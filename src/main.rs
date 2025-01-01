use std::{env, process::ExitCode};

use kestrel::{Database, Query, SearchOptions};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kestrel: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> kestrel::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage:\n  kestrel put <db-dir> <rowid> <text>\n  kestrel get <db-dir> <rowid>\n  kestrel search <db-dir> <term> [limit]\n  kestrel optimize <db-dir>"
        );
        return Ok(());
    }
    match args[1].as_str() {
        "put" if args.len() >= 5 => {
            let db = Database::open(&args[2])?;
            let rowid = parse_rowid(&args[3])?;
            let mut tx = db.begin();
            tx.upsert_text(rowid, args[4..].join(" "))?;
            let generation = tx.commit()?;
            println!("committed generation {generation}");
        }
        "get" if args.len() == 4 => {
            let db = Database::open(&args[2])?;
            let rowid = parse_rowid(&args[3])?;
            if let Some(fields) = db.get(rowid) {
                println!("{}", fields.join("\t"));
            }
        }
        "search" if args.len() >= 4 => {
            let db = Database::open(&args[2])?;
            let limit = args
                .get(4)
                .map(|value| value.parse())
                .transpose()
                .map_err(|_| kestrel::Error::InvalidInput("limit must be an integer".to_owned()))?
                .unwrap_or(10);
            let result = db.search(
                &Query::term(&args[3]),
                SearchOptions {
                    limit,
                    ..SearchOptions::default()
                },
            )?;
            for hit in result.hits {
                println!("{}\t{:.6}\t{}", hit.rowid, hit.score, hit.fields.join("\t"));
            }
        }
        "optimize" if args.len() == 3 => {
            Database::open(&args[2])?.optimize()?;
            println!("checkpoint written");
        }
        _ => {
            return Err(kestrel::Error::InvalidInput(
                "invalid command or arguments".to_owned(),
            ));
        }
    }
    Ok(())
}

fn parse_rowid(value: &str) -> kestrel::Result<i64> {
    value
        .parse()
        .map_err(|_| kestrel::Error::InvalidInput("rowid must be an i64".to_owned()))
}
