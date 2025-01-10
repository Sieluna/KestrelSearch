use std::{
    fs,
    hint::black_box,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use kestrel::{Database, DatabaseConfig, Query, SearchOptions, SearchResult};

fn main() -> kestrel::Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("kestrel-bench-{}-{nonce}", std::process::id()));
    let db = Database::open_with_config(
        &path,
        DatabaseConfig {
            query_cache_capacity: 0,
        },
    )?;

    const DOCS: i64 = 100_000;
    const BATCH: i64 = 5_000;
    let build_start = Instant::now();
    for base in (0..DOCS).step_by(BATCH as usize) {
        let mut tx = db.begin();
        for rowid in base..(base + BATCH).min(DOCS) {
            let mut text = String::from("database embedded common search ");
            if rowid % 7 == 0 {
                text.push_str("rust engine ");
            }
            if rowid % 997 == 0 {
                text.push_str("kestrel kestrel kestrel rare ");
            }
            tx.upsert_text(rowid, text)?;
        }
        tx.commit()?;
    }
    let build_time = build_start.elapsed();

    let options = SearchOptions {
        limit: 10,
        cache: false,
        ..Default::default()
    };
    println!("documents:       {DOCS}");
    println!("build:           {}", format_duration(build_time));
    println!("workload                 mean       candidates  scored  skipped");
    bench_query(
        &db,
        "or-common",
        Query::or([
            Query::term("kestrel"),
            Query::term("rust"),
            Query::term("common"),
        ]),
        options,
    )?;
    bench_query(&db, "rare-term", Query::term("kestrel"), options)?;
    bench_query(
        &db,
        "and-common",
        Query::and([Query::term("common"), Query::term("rust")]),
        options,
    )?;
    bench_query(
        &db,
        "phrase-common",
        Query::phrase(["database", "embedded"]),
        options,
    )?;
    bench_query(&db, "prefix", Query::prefix("eng"), options)?;

    drop(db);
    let _ = fs::remove_dir_all(path);
    Ok(())
}

fn bench_query(
    db: &Database,
    name: &str,
    query: Query,
    options: SearchOptions,
) -> kestrel::Result<()> {
    const ITERATIONS: u32 = 100;
    for _ in 0..10 {
        black_box(db.search(&query, options)?);
    }
    let started = Instant::now();
    let mut last: Option<SearchResult> = None;
    for _ in 0..ITERATIONS {
        last = Some(black_box(db.search(&query, options)?));
    }
    let elapsed = started.elapsed() / ITERATIONS;
    let stats = last.unwrap().stats;
    println!(
        "{name:<22} {:>10} {:>12} {:>7} {:>8}",
        format_duration(elapsed),
        stats.candidate_docs,
        stats.scored_docs,
        stats.skipped_blocks,
    );
    Ok(())
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.3} s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3} us", duration.as_secs_f64() * 1_000_000.0)
    }
}
