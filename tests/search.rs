use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Barrier},
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
fn prefix_ranges_across_segments_ignore_terms_with_no_live_documents() {
    let path = temp_dir("prefix-segment-ranges");
    let db = Database::open(&path).unwrap();
    let mut first = db.begin();
    first.upsert_text(1, "apple obsolete").unwrap();
    first.commit().unwrap();
    let mut second = db.begin();
    second.upsert_text(1, "banana replacement").unwrap();
    second.upsert_text(2, "application live").unwrap();
    second.commit().unwrap();

    assert_eq!(ids(&db, &Query::prefix("app")), [2]);
    drop(db);
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

#[test]
fn concurrent_writers_publish_prebuilt_segments_without_lost_updates() {
    let path = temp_dir("concurrent-writers");
    let db = Arc::new(Database::open(&path).unwrap());
    let barrier = Arc::new(Barrier::new(5));
    let writers: Vec<_> = (0..4_i64)
        .map(|writer| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut tx = db.begin();
                for offset in 0..200_i64 {
                    let rowid = writer * 1_000 + offset;
                    tx.upsert_text(
                        rowid,
                        format!("writer{writer} shared prebuilt term{}", offset % 19),
                    )
                    .unwrap();
                }
                barrier.wait();
                tx.commit().unwrap()
            })
        })
        .collect();
    barrier.wait();
    let mut generations: Vec<_> = writers
        .into_iter()
        .map(|writer| writer.join().unwrap())
        .collect();
    generations.sort_unstable();
    assert_eq!(generations, [1, 2, 3, 4]);
    assert_eq!(db.len(), 800);
    for writer in 0..4 {
        assert_eq!(ids(&db, &Query::term(format!("writer{writer}"))).len(), 100);
    }
    drop(db);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn randomized_pruning_matches_exhaustive_across_segments_and_updates() {
    let path = temp_dir("random-differential");
    let db = Database::open(&path).unwrap();
    let mut state = 0x4d59_5df4_d0f3_3173_u64;

    for batch in 0..6_i64 {
        let mut tx = db.begin();
        for offset in 0..200_i64 {
            let rowid = batch * 200 + offset;
            tx.upsert_text(rowid, random_document(&mut state)).unwrap();
        }
        tx.commit().unwrap();
    }
    let mut mutations = db.begin();
    for rowid in (0..180_i64).step_by(3) {
        mutations
            .upsert_text(rowid, random_document(&mut state))
            .unwrap();
    }
    for rowid in (1..180_i64).step_by(7) {
        mutations.delete(rowid).unwrap();
    }
    mutations.commit().unwrap();

    for round in 0..200 {
        let left = format!("t{:02}", next_random(&mut state) % 32);
        let right = format!("t{:02}", next_random(&mut state) % 32);
        let query = match round % 6 {
            0 => Query::term(&left),
            1 => Query::or([Query::term(&left), Query::term(&right)]),
            2 => Query::and([Query::term(&left), Query::term(&right)]),
            3 => Query::phrase([&left, &right]),
            4 => Query::prefix(format!("t{}", next_random(&mut state) % 3)),
            _ => Query::and([Query::term(&left), Query::negate(Query::term(&right))]),
        };
        let options = SearchOptions {
            limit: (next_random(&mut state) as usize % 20) + 1,
            cache: false,
            ..Default::default()
        };
        let fast = db.search(&query, options).unwrap();
        let exhaustive = db.search_exhaustive(&query, options).unwrap();
        assert_eq!(
            fast.hits
                .iter()
                .map(|hit| (hit.rowid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            exhaustive
                .hits
                .iter()
                .map(|hit| (hit.rowid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            "query {query:?}"
        );
    }
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn equal_scores_keep_smallest_rowids_across_commit_order() {
    let path = temp_dir("ties");
    let db = Database::open(&path).unwrap();
    for rowids in [(100..200).collect::<Vec<_>>(), (0..100).collect()] {
        let mut tx = db.begin();
        for rowid in rowids {
            tx.upsert_text(rowid, "identical").unwrap();
        }
        tx.commit().unwrap();
    }
    let result = db
        .search(
            &Query::term("identical"),
            SearchOptions {
                limit: 10,
                cache: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        result.hits.iter().map(|hit| hit.rowid).collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn native_checkpoint_preserves_exact_queries_after_reopen() {
    let path = temp_dir("native-reopen-differential");
    let db = Database::open(&path).unwrap();
    for batch in 0..5_i64 {
        let mut tx = db.begin();
        for offset in 0..240_i64 {
            let rowid = batch * 240 + offset;
            let text = format!(
                "common t{:02} t{:02} ordered phrase{}",
                rowid % 29,
                (rowid * 7) % 29,
                rowid % 13
            );
            tx.upsert_text(rowid, text).unwrap();
        }
        tx.commit().unwrap();
    }
    let mut changes = db.begin();
    for rowid in (0..300_i64).step_by(11) {
        changes
            .upsert_text(rowid, format!("replacement t{:02} exact", rowid % 29))
            .unwrap();
    }
    for rowid in (3..300_i64).step_by(17) {
        changes.delete(rowid).unwrap();
    }
    changes.commit().unwrap();

    let queries = [
        Query::term("common"),
        Query::or([Query::term("t03"), Query::term("t17")]),
        Query::and([Query::term("common"), Query::term("t09")]),
        Query::phrase(["ordered", "phrase4"]),
        Query::prefix("t2"),
        Query::near(["common", "ordered"], 3),
    ];
    let options = SearchOptions {
        limit: 37,
        cache: false,
        ..Default::default()
    };
    let expected: Vec<_> = queries
        .iter()
        .map(|query| {
            db.search(query, options)
                .unwrap()
                .hits
                .into_iter()
                .map(|hit| (hit.rowid, hit.score.to_bits(), hit.fields))
                .collect::<Vec<_>>()
        })
        .collect();
    db.optimize().unwrap();
    drop(db);

    let reopened = Database::open(&path).unwrap();
    for (query, expected) in queries.iter().zip(expected) {
        let actual = reopened
            .search(query, options)
            .unwrap()
            .hits
            .into_iter()
            .map(|hit| (hit.rowid, hit.score.to_bits(), hit.fields))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "query {query:?}");
    }
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

fn random_document(state: &mut u64) -> String {
    let terms = 5 + (next_random(state) % 20);
    (0..terms)
        .map(|_| format!("t{:02}", next_random(state) % 32))
        .collect::<Vec<_>>()
        .join(" ")
}

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
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
