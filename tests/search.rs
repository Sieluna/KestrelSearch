use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use kestrel::{Database, Query, SearchOptions};

fn temp_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "kestrel-integration-{name}-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn typed_boolean_phrase_prefix_and_delete() {
    let path = temp_dir("queries");
    let db = Database::open(&path).unwrap();
    let mut tx = db.begin();
    tx.upsert(1, ["Rust search", "fast embedded database"])
        .unwrap();
    tx.upsert(2, ["Rust database", "full text engine"]).unwrap();
    tx.upsert(3, ["Other", "embedded search system"]).unwrap();
    tx.commit().unwrap();

    let and = Query::and([Query::term("rust"), Query::term("database")]);
    assert_eq!(ids(&db, &and), [1, 2]);
    assert_eq!(ids(&db, &Query::phrase(["embedded", "database"])), [1]);
    assert_eq!(ids(&db, &Query::prefix("eng")), [2]);
    assert_eq!(
        ids(
            &db,
            &Query::and([Query::term("rust"), Query::negate(Query::term("engine"))])
        ),
        [1]
    );

    let mut tx = db.begin();
    tx.delete(1).unwrap();
    tx.upsert_text(2, "replacement without old terms").unwrap();
    tx.commit().unwrap();
    assert!(ids(&db, &Query::term("rust")).is_empty());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn cjk_bigrams_are_searchable() {
    let path = temp_dir("cjk");
    let db = Database::open(&path).unwrap();
    let mut tx = db.begin();
    tx.upsert_text(7, "这是全文检索数据库").unwrap();
    tx.commit().unwrap();
    assert_eq!(ids(&db, &Query::term("全文")), [7]);
    assert_eq!(ids(&db, &Query::phrase(["全文", "文检", "检索"])), [7]);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn column_and_near_queries_verify_positions_after_candidate_generation() {
    let path = temp_dir("positions");
    let db = Database::open(&path).unwrap();
    let mut tx = db.begin();
    tx.upsert(1, ["rust engine", "embedded database search"])
        .unwrap();
    tx.upsert(2, ["database", "rust fast engine search"])
        .unwrap();
    tx.upsert(3, ["rust", "engine separated across columns"])
        .unwrap();
    tx.upsert(4, ["rust once", "nothing nearby"]).unwrap();
    tx.upsert(5, ["rust rust", "two occurrences"]).unwrap();
    tx.commit().unwrap();

    assert_eq!(ids(&db, &Query::column(0, Query::term("engine"))), [1]);
    assert_eq!(ids(&db, &Query::column(1, Query::term("engine"))), [2, 3]);
    assert_eq!(ids(&db, &Query::near(["rust", "engine"], 2)), [1, 2]);
    assert_eq!(
        ids(&db, &Query::column(1, Query::near(["rust", "engine"], 2))),
        [2]
    );
    assert_eq!(ids(&db, &Query::near(["rust", "rust"], 1)), [5]);
    assert!(
        ids(
            &db,
            &Query::column(0, Query::column(1, Query::term("engine")))
        )
        .is_empty()
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn pruned_results_equal_exhaustive_results() {
    let path = temp_dir("differential");
    let db = Database::open(&path).unwrap();
    for batch in 0..8 {
        let mut tx = db.begin();
        for i in 0..180 {
            let id = batch * 180 + i;
            let mut text = String::from("common ");
            if id % 2 == 0 {
                text.push_str("even ");
            }
            if id % 3 == 0 {
                text.push_str("third ");
            }
            if id % 17 == 0 {
                text.push_str("rare rare rare ");
            }
            text.push_str(&format!("group{}", id % 11));
            tx.upsert_text(id, text).unwrap();
        }
        tx.commit().unwrap();
    }
    let queries = [
        Query::term("common"),
        Query::or([Query::term("rare"), Query::term("third")]),
        Query::and([Query::term("common"), Query::term("even")]),
        Query::prefix("group"),
    ];
    for query in queries {
        for limit in [1, 3, 10, 50] {
            let options = SearchOptions {
                limit,
                cache: false,
                ..Default::default()
            };
            let fast = db.search(&query, options).unwrap();
            let exact = db.search_exhaustive(&query, options).unwrap();
            assert_eq!(
                fast.hits
                    .iter()
                    .map(|hit| (hit.rowid, hit.score.to_bits()))
                    .collect::<Vec<_>>(),
                exact
                    .hits
                    .iter()
                    .map(|hit| (hit.rowid, hit.score.to_bits()))
                    .collect::<Vec<_>>()
            );
        }
    }

    let measured = db
        .search(
            &Query::or([
                Query::term("rare"),
                Query::term("third"),
                Query::term("common"),
            ]),
            SearchOptions {
                limit: 10,
                cache: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(measured.stats.skipped_blocks > 0);
    assert!(measured.stats.scored_docs < db.len() as usize);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn cache_is_generation_scoped_and_invalidated() {
    let path = temp_dir("cache");
    let db = Database::open(&path).unwrap();
    let mut tx = db.begin();
    tx.upsert_text(1, "cached query").unwrap();
    tx.commit().unwrap();
    let query = Query::term("cached");
    assert!(
        !db.search(&query, SearchOptions::default())
            .unwrap()
            .stats
            .cache_hit
    );
    assert!(
        db.search(&query, SearchOptions::default())
            .unwrap()
            .stats
            .cache_hit
    );
    let mut tx = db.begin();
    tx.upsert_text(2, "cached again").unwrap();
    tx.commit().unwrap();
    assert!(
        !db.search(&query, SearchOptions::default())
            .unwrap()
            .stats
            .cache_hit
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn concurrent_readers_share_sharded_cache() {
    let path = temp_dir("concurrent-cache");
    let db = Arc::new(Database::open(&path).unwrap());
    let mut tx = db.begin();
    for rowid in 0..500 {
        tx.upsert_text(rowid, format!("shared cache term{}", rowid % 17))
            .unwrap();
    }
    tx.commit().unwrap();

    let readers: Vec<_> = (0..8)
        .map(|reader| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                for round in 0..100 {
                    let query = Query::term(format!("term{}", (reader + round) % 17));
                    let result = db.search(&query, SearchOptions::default()).unwrap();
                    assert_eq!(result.generation, 1);
                }
            })
        })
        .collect();
    for reader in readers {
        reader.join().unwrap();
    }
    drop(db);
    fs::remove_dir_all(path).unwrap();
}

fn ids(db: &Database, query: &Query) -> Vec<i64> {
    let mut ids: Vec<_> = db
        .search(
            query,
            SearchOptions {
                limit: 100,
                ..Default::default()
            },
        )
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.rowid)
        .collect();
    ids.sort_unstable();
    ids
}
