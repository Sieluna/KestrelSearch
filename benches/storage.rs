use std::{
    fs,
    hint::black_box,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use kestrel::{Database, DatabaseConfig, Query, SearchOptions};

fn main() -> kestrel::Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "kestrel-storage-bench-{}-{nonce}",
        std::process::id()
    ));
    const DOCS: i64 = 100_000;
    const BATCH: i64 = 5_000;

    let db = Database::open_with_config(
        &path,
        DatabaseConfig {
            query_cache_capacity: 0,
        },
    )?;
    let build_started = Instant::now();
    for base in (0..DOCS).step_by(BATCH as usize) {
        let mut tx = db.begin();
        for rowid in base..(base + BATCH).min(DOCS) {
            tx.upsert_text(
                rowid,
                format!(
                    "common database embedded search rust group{} unique{rowid}",
                    rowid % 1_009
                ),
            )?;
        }
        tx.commit()?;
    }
    let build = build_started.elapsed();

    let optimize_started = Instant::now();
    db.optimize()?;
    let optimize = optimize_started.elapsed();
    let checkpoint_bytes = fs::read_dir(&path)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "kst"))
        .map(fs::metadata)
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|metadata| metadata.len())
        .sum::<u64>();
    drop(db);

    let reopen_started = Instant::now();
    let reopened = Database::open_with_config(
        &path,
        DatabaseConfig {
            query_cache_capacity: 0,
        },
    )?;
    let reopen = reopen_started.elapsed();
    let query = Query::or([
        Query::term("unique99991"),
        Query::term("group17"),
        Query::term("common"),
    ]);
    let options = SearchOptions {
        cache: false,
        ..SearchOptions::default()
    };
    let first_query_started = Instant::now();
    black_box(reopened.search(&query, options)?);
    let first_query = first_query_started.elapsed();

    println!("documents:       {DOCS}");
    println!("build:           {}", format_duration(build));
    println!("optimize:        {}", format_duration(optimize));
    println!(
        "checkpoint:      {:.3} MiB",
        checkpoint_bytes as f64 / 1_048_576.0
    );
    println!(
        "bytes/document:  {:.2}",
        checkpoint_bytes as f64 / DOCS as f64
    );
    println!("reopen:          {}", format_duration(reopen));
    println!("first query:     {}", format_duration(first_query));

    drop(reopened);
    let _ = fs::remove_dir_all(path);
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
