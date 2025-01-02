use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    Error, Query, Result, SearchOptions, SearchResult,
    codec::{
        checksum, put_bytes, put_i64, put_u32, put_u64, read_bytes, read_i64, read_u32, read_u64,
    },
    index::Snapshot,
};

const WAL_MAGIC: &[u8; 8] = b"KSTWAL01";
const SEGMENT_MAGIC: &[u8; 8] = b"KSTSEG01";
const MAX_RECORD_BYTES: usize = 256 * 1024 * 1024;

type WalCommit = (u64, Vec<Operation>);
type WalContents = (Vec<WalCommit>, u64);

#[derive(Clone, Copy, Debug)]
pub struct DatabaseConfig {
    /// Number of generation-scoped top-k results retained in the in-process LRU.
    pub query_cache_capacity: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            query_cache_capacity: 256,
        }
    }
}

#[derive(Clone)]
pub struct Database {
    inner: Arc<Inner>,
}

struct Inner {
    path: PathBuf,
    snapshot: RwLock<Arc<Snapshot>>,
    wal: Mutex<File>,
    writer: Mutex<()>,
    cache: Mutex<QueryCache>,
    checkpoint_nonce: AtomicU64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Operation {
    Upsert { rowid: i64, fields: Vec<String> },
    Delete { rowid: i64 },
}

pub struct Transaction {
    database: Database,
    operations: Option<Vec<Operation>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    generation: u64,
    query: Query,
    limit: usize,
    k1: u32,
    b: u32,
}

struct QueryCache {
    capacity: usize,
    entries: HashMap<CacheKey, Arc<SearchResult>>,
    order: VecDeque<CacheKey>,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(path, DatabaseConfig::default())
    }

    pub fn open_with_config(path: impl AsRef<Path>, config: DatabaseConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let mut snapshot = load_latest_checkpoint(&path)?.unwrap_or_default();
        let wal_path = path.join("kestrel.wal");
        let mut wal = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&wal_path)?;
        let (commits, valid_len) = read_wal(&mut wal)?;
        if wal.metadata()?.len() != valid_len {
            wal.set_len(valid_len)?;
        }
        wal.seek(SeekFrom::End(0))?;

        for (generation, operations) in commits {
            if generation > snapshot.generation {
                snapshot = apply_operations(&snapshot, generation, operations);
            }
        }

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Ok(Self {
            inner: Arc::new(Inner {
                path,
                snapshot: RwLock::new(Arc::new(snapshot)),
                wal: Mutex::new(wal),
                writer: Mutex::new(()),
                cache: Mutex::new(QueryCache::new(config.query_cache_capacity)),
                checkpoint_nonce: AtomicU64::new(seed),
            }),
        })
    }

    pub fn begin(&self) -> Transaction {
        Transaction {
            database: self.clone(),
            operations: Some(Vec::new()),
        }
    }

    pub fn search(&self, query: &Query, options: SearchOptions) -> Result<SearchResult> {
        let options = options.validate()?;
        let snapshot = self.snapshot();
        let key = CacheKey {
            generation: snapshot.generation,
            query: query.clone(),
            limit: options.limit,
            k1: options.k1.to_bits(),
            b: options.b.to_bits(),
        };
        if options.cache
            && let Some(mut result) = self.inner.cache.lock().unwrap().get(&key)
        {
            result.stats.cache_hit = true;
            return Ok(result);
        }

        let result = snapshot.search(query, options, false);
        if options.cache {
            self.inner
                .cache
                .lock()
                .unwrap()
                .insert(key, Arc::new(result.clone()));
        }
        Ok(result)
    }

    /// Reference path used by differential tests. It disables block pruning and caching.
    pub fn search_exhaustive(
        &self,
        query: &Query,
        mut options: SearchOptions,
    ) -> Result<SearchResult> {
        options.cache = false;
        let options = options.validate()?;
        Ok(self.snapshot().search(query, options, true))
    }

    pub fn get(&self, rowid: i64) -> Option<Vec<String>> {
        let snapshot = self.snapshot();
        let location = snapshot.latest.get(&rowid)?;
        if location.deleted {
            return None;
        }
        Some(
            snapshot.segments[location.segment].docs[location.docid as usize]
                .fields
                .clone(),
        )
    }

    pub fn generation(&self) -> u64 {
        self.snapshot().generation
    }

    pub fn len(&self) -> u64 {
        self.snapshot().live_docs
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes one immutable, checksummed checkpoint containing only live documents,
    /// then publishes the compacted in-memory segment. The append-only WAL remains
    /// an audit/recovery log; records at or below the checkpoint generation are skipped.
    pub fn optimize(&self) -> Result<()> {
        let _writer = self.inner.writer.lock().unwrap();
        let current = self.snapshot();
        let compacted = current.compacted();
        let mut operations: Vec<_> = compacted
            .latest
            .values()
            .filter(|location| !location.deleted)
            .map(|location| {
                let doc = &compacted.segments[location.segment].docs[location.docid as usize];
                Operation::Upsert {
                    rowid: doc.rowid,
                    fields: doc.fields.clone(),
                }
            })
            .collect();
        operations.sort_by_key(|operation| match operation {
            Operation::Upsert { rowid, .. } | Operation::Delete { rowid } => *rowid,
        });
        let payload = encode_commit(compacted.generation, &operations)?;
        let nonce = self.inner.checkpoint_nonce.fetch_add(1, Ordering::Relaxed);
        let temp = self
            .inner
            .path
            .join(format!("segment-{}-{nonce}.tmp", compacted.generation));
        let final_path = self
            .inner
            .path
            .join(format!("segment-{}-{nonce}.kst", compacted.generation));
        write_framed_file(&temp, SEGMENT_MAGIC, &payload)?;
        fs::rename(&temp, &final_path)?;
        *self.inner.snapshot.write().unwrap() = Arc::new(compacted);
        self.inner.cache.lock().unwrap().clear();
        Ok(())
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&self.inner.snapshot.read().unwrap())
    }

    fn commit(&self, operations: Vec<Operation>) -> Result<u64> {
        if operations.is_empty() {
            return Ok(self.generation());
        }
        validate_operations(&operations)?;
        let _writer = self.inner.writer.lock().unwrap();
        let current = self.snapshot();
        let generation = current
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("generation counter exhausted".to_owned()))?;
        let payload = encode_commit(generation, &operations)?;
        let frame = encode_frame(WAL_MAGIC, &payload)?;

        {
            let mut wal = self.inner.wal.lock().unwrap();
            wal.seek(SeekFrom::End(0))?;
            wal.write_all(&frame)?;
            wal.sync_data()?;
        }

        let next = apply_operations(&current, generation, operations);
        *self.inner.snapshot.write().unwrap() = Arc::new(next);
        self.inner.cache.lock().unwrap().clear();
        Ok(generation)
    }
}

impl Transaction {
    pub fn upsert(
        &mut self,
        rowid: i64,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<&mut Self> {
        let operations = self.operations.as_mut().ok_or(Error::TransactionClosed)?;
        operations.push(Operation::Upsert {
            rowid,
            fields: fields.into_iter().map(Into::into).collect(),
        });
        Ok(self)
    }

    pub fn upsert_text(&mut self, rowid: i64, text: impl Into<String>) -> Result<&mut Self> {
        self.upsert(rowid, [text.into()])
    }

    pub fn delete(&mut self, rowid: i64) -> Result<&mut Self> {
        let operations = self.operations.as_mut().ok_or(Error::TransactionClosed)?;
        operations.push(Operation::Delete { rowid });
        Ok(self)
    }

    pub fn commit(mut self) -> Result<u64> {
        let operations = self.operations.take().ok_or(Error::TransactionClosed)?;
        self.database.commit(operations)
    }

    pub fn rollback(mut self) -> Result<()> {
        self.operations.take().ok_or(Error::TransactionClosed)?;
        Ok(())
    }
}

impl QueryCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<SearchResult> {
        let result = self.entries.get(key)?.as_ref().clone();
        if let Some(index) = self.order.iter().position(|candidate| candidate == key) {
            self.order.remove(index);
        }
        self.order.push_back(key.clone());
        Some(result)
    }

    fn insert(&mut self, key: CacheKey, value: Arc<SearchResult>) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.insert(key.clone(), value).is_some()
            && let Some(index) = self.order.iter().position(|candidate| *candidate == key)
        {
            self.order.remove(index);
        }
        self.order.push_back(key);
        while self.entries.len() > self.capacity {
            if let Some(expired) = self.order.pop_front() {
                self.entries.remove(&expired);
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

fn validate_operations(operations: &[Operation]) -> Result<()> {
    for operation in operations {
        if let Operation::Upsert { fields, .. } = operation {
            if fields.is_empty() {
                return Err(Error::InvalidInput(
                    "a document requires at least one field".to_owned(),
                ));
            }
            if fields.len() > 64 {
                return Err(Error::InvalidInput(
                    "a document supports at most 64 fields".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn apply_operations(snapshot: &Snapshot, generation: u64, operations: Vec<Operation>) -> Snapshot {
    // Last operation for a rowid wins inside a transaction.
    let mut last = HashMap::<i64, Operation>::new();
    for operation in operations {
        let rowid = match &operation {
            Operation::Upsert { rowid, .. } | Operation::Delete { rowid } => *rowid,
        };
        last.insert(rowid, operation);
    }
    let mut ordered: Vec<_> = last.into_values().collect();
    ordered.sort_by_key(|operation| match operation {
        Operation::Upsert { rowid, .. } | Operation::Delete { rowid } => *rowid,
    });
    let mut docs = Vec::new();
    let mut deletes = Vec::new();
    for operation in ordered {
        match operation {
            Operation::Upsert { rowid, fields } => docs.push((rowid, fields)),
            Operation::Delete { rowid } => deletes.push(rowid),
        }
    }
    snapshot.with_commit(generation, docs, deletes)
}

fn encode_commit(generation: u64, operations: &[Operation]) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    put_u64(&mut payload, generation);
    put_u32(
        &mut payload,
        u32::try_from(operations.len()).map_err(|_| {
            Error::InvalidInput("too many operations in one transaction".to_owned())
        })?,
    );
    for operation in operations {
        match operation {
            Operation::Upsert { rowid, fields } => {
                payload.push(1);
                put_i64(&mut payload, *rowid);
                put_u32(&mut payload, fields.len() as u32);
                for field in fields {
                    put_bytes(&mut payload, field.as_bytes())?;
                }
            }
            Operation::Delete { rowid } => {
                payload.push(2);
                put_i64(&mut payload, *rowid);
            }
        }
    }
    Ok(payload)
}

fn decode_commit(mut payload: &[u8]) -> Result<(u64, Vec<Operation>)> {
    let generation = read_u64(&mut payload)?;
    let count = read_u32(&mut payload)? as usize;
    let mut operations = Vec::with_capacity(count.min(1_000_000));
    for _ in 0..count {
        let (&tag, rest) = payload
            .split_first()
            .ok_or(Error::Corrupt("missing operation tag"))?;
        payload = rest;
        let rowid = read_i64(&mut payload)?;
        match tag {
            1 => {
                let field_count = read_u32(&mut payload)? as usize;
                if field_count == 0 || field_count > 64 {
                    return Err(Error::Corrupt("invalid field count"));
                }
                let mut fields = Vec::with_capacity(field_count);
                for _ in 0..field_count {
                    let bytes = read_bytes(&mut payload)?;
                    fields.push(
                        String::from_utf8(bytes.to_vec())
                            .map_err(|_| Error::Corrupt("field is not UTF-8"))?,
                    );
                }
                operations.push(Operation::Upsert { rowid, fields });
            }
            2 => operations.push(Operation::Delete { rowid }),
            _ => return Err(Error::Corrupt("unknown operation tag")),
        }
    }
    if !payload.is_empty() {
        return Err(Error::Corrupt("trailing commit bytes"));
    }
    Ok((generation, operations))
}

fn encode_frame(magic: &[u8; 8], payload: &[u8]) -> Result<Vec<u8>> {
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::InvalidInput("transaction exceeds 4 GiB".to_owned()))?;
    let mut frame = Vec::with_capacity(16 + payload.len());
    frame.extend_from_slice(magic);
    put_u32(&mut frame, len);
    put_u32(&mut frame, checksum(payload));
    frame.extend_from_slice(payload);
    Ok(frame)
}

fn write_framed_file(path: &Path, magic: &[u8; 8], payload: &[u8]) -> Result<()> {
    let frame = encode_frame(magic, payload)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&frame)?;
    file.sync_all()?;
    Ok(())
}

fn read_wal(file: &mut File) -> Result<WalContents> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let mut offset = 0_usize;
    let mut valid = 0_usize;
    let mut commits = Vec::new();
    let mut previous_generation = 0_u64;
    while offset < bytes.len() {
        if bytes.len() - offset < 16 {
            break;
        }
        if &bytes[offset..offset + 8] != WAL_MAGIC {
            return Err(Error::Corrupt("invalid WAL magic"));
        }
        let len = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap()) as usize;
        let expected = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
        if len > MAX_RECORD_BYTES {
            return Err(Error::Corrupt("WAL record exceeds safety limit"));
        }
        let end = offset + 16 + len;
        if end > bytes.len() {
            break;
        }
        let payload = &bytes[offset + 16..end];
        if checksum(payload) != expected {
            if end == bytes.len() {
                break;
            }
            return Err(Error::Corrupt("WAL checksum mismatch"));
        }
        let (generation, operations) = decode_commit(payload)?;
        if generation <= previous_generation {
            return Err(Error::Corrupt("WAL generations are not increasing"));
        }
        previous_generation = generation;
        commits.push((generation, operations));
        offset = end;
        valid = end;
    }
    Ok((commits, valid as u64))
}

fn load_latest_checkpoint(path: &Path) -> Result<Option<Snapshot>> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with("segment-") && file_name.ends_with(".kst") {
            candidates.push(entry.path());
        }
    }

    let mut best: Option<Snapshot> = None;
    for candidate in candidates {
        let Ok((generation, operations)) = read_checkpoint(&candidate) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| generation > current.generation)
        {
            best = Some(apply_operations(
                &Snapshot::default(),
                generation,
                operations,
            ));
        }
    }
    Ok(best)
}

fn read_checkpoint(path: &Path) -> Result<(u64, Vec<Operation>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 16 || &bytes[..8] != SEGMENT_MAGIC {
        return Err(Error::Corrupt("invalid checkpoint header"));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let expected = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if len > MAX_RECORD_BYTES || bytes.len() != 16 + len {
        return Err(Error::Corrupt("invalid checkpoint length"));
    }
    let payload = &bytes[16..];
    if checksum(payload) != expected {
        return Err(Error::Corrupt("checkpoint checksum mismatch"));
    }
    decode_commit(payload)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::Query;

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kestrel-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn transaction_reopens_and_rollback_is_invisible() {
        let path = temp_dir("reopen");
        {
            let db = Database::open(&path).unwrap();
            let mut tx = db.begin();
            tx.upsert_text(10, "fast embedded search").unwrap();
            tx.upsert_text(20, "durable database").unwrap();
            assert_eq!(tx.commit().unwrap(), 1);
            let mut rolled_back = db.begin();
            rolled_back.upsert_text(30, "never visible").unwrap();
            rolled_back.rollback().unwrap();
            assert_eq!(db.len(), 2);
        }
        let db = Database::open(&path).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(
            db.search(&Query::term("search"), SearchOptions::default())
                .unwrap()
                .hits[0]
                .rowid,
            10
        );
        assert!(db.get(30).is_none());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn torn_wal_tail_is_discarded() {
        let path = temp_dir("torn");
        let db = Database::open(&path).unwrap();
        let mut tx = db.begin();
        tx.upsert_text(1, "committed").unwrap();
        tx.commit().unwrap();
        drop(db);
        let committed_len = fs::metadata(path.join("kestrel.wal")).unwrap().len();
        OpenOptions::new()
            .append(true)
            .open(path.join("kestrel.wal"))
            .unwrap()
            .write_all(b"KST")
            .unwrap();
        let db = Database::open(&path).unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(
            fs::metadata(path.join("kestrel.wal")).unwrap().len(),
            committed_len
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn checkpoint_restores_compacted_view() {
        let path = temp_dir("checkpoint");
        {
            let db = Database::open(&path).unwrap();
            let mut tx = db.begin();
            tx.upsert_text(1, "old value").unwrap();
            tx.commit().unwrap();
            let mut tx = db.begin();
            tx.upsert_text(1, "new value").unwrap();
            tx.upsert_text(2, "keep me").unwrap();
            tx.commit().unwrap();
            db.optimize().unwrap();
        }
        let db = Database::open(&path).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.get(1).unwrap(), ["new value"]);
        fs::remove_dir_all(path).unwrap();
    }
}
