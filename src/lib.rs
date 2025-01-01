//! Kestrel is a dependency-free, embedded full-text database.
//!
//! Its public query API is typed and bypasses SQL and textual query parsing.
//! Writes are transactional and durable through a checksummed append-only WAL;
//! readers clone a short-lived immutable snapshot pointer and never hold a lock
//! while evaluating a query.

mod codec;
mod error;
mod index;
mod query;
mod storage;
mod tokenizer;

pub use error::{Error, Result};
pub use query::{Query, SearchHit, SearchOptions, SearchResult, SearchStats};
pub use storage::{Database, DatabaseConfig, Transaction};
