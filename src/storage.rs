use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File, OpenOptions},
    hash::{DefaultHasher, Hash, Hasher},
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
    index::{BlockMeta, Location, PostingList, Segment, Snapshot, StoredDocument},
};

const WAL_MAGIC: &[u8; 8] = b"KSTWAL01";
const LEGACY_SEGMENT_MAGIC: &[u8; 8] = b"KSTSEG01";
const SEGMENT_MAGIC: &[u8; 8] = b"KSTSEG02";
const MAX_WAL_RECORD_BYTES: usize = 256 * 1024 * 1024;

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
    cache: CacheShards,
    checkpoint_nonce: AtomicU64,
}

struct PreparedCommit {
    wal_payload: Vec<u8>,
    segment: Option<Arc<Segment>>,
    deletes: Vec<i64>,
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

struct CacheShards {
    shards: Box<[Mutex<QueryCache>]>,
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
                cache: CacheShards::new(config.query_cache_capacity),
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
            && let Some(mut result) = self.inner.cache.get(&key)
        {
            result.stats.cache_hit = true;
            return Ok(result);
        }

        let result = snapshot.search(query, options, false);
        if options.cache {
            self.inner.cache.insert(key, Arc::new(result.clone()));
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

    /// Writes one immutable, checksummed, query-ready checkpoint containing only
    /// live documents, then publishes the compacted in-memory segment.
    pub fn optimize(&self) -> Result<()> {
        let _writer = self.inner.writer.lock().unwrap();
        let current = self.snapshot();
        let compacted = current.compacted();
        let payload = encode_snapshot(&compacted)?;
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

        // The durable checkpoint now covers every generation in the WAL. Rotate
        // only after the rename so a crash always leaves at least one source of truth.
        {
            let mut wal = self.inner.wal.lock().unwrap();
            wal.set_len(0)?;
            wal.seek(SeekFrom::Start(0))?;
            wal.sync_all()?;
        }
        *self.inner.snapshot.write().unwrap() = Arc::new(compacted);
        self.inner.cache.clear();
        remove_old_checkpoints(&self.inner.path, &final_path)?;
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
        // Canonicalization, WAL encoding, tokenization, and posting construction
        // do not depend on the current snapshot. Keep them outside the serialized
        // durability/publication section so concurrent writers only contend on I/O.
        let mut prepared = prepare_commit(operations)?;
        let _writer = self.inner.writer.lock().unwrap();
        let current = self.snapshot();
        let generation = current
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("generation counter exhausted".to_owned()))?;
        prepared.wal_payload[..8].copy_from_slice(&generation.to_le_bytes());
        let frame = encode_frame(WAL_MAGIC, &prepared.wal_payload)?;

        {
            let mut wal = self.inner.wal.lock().unwrap();
            wal.seek(SeekFrom::End(0))?;
            wal.write_all(&frame)?;
            wal.sync_data()?;
        }

        let next = current.with_prebuilt_commit(generation, prepared.segment, prepared.deletes);
        *self.inner.snapshot.write().unwrap() = Arc::new(next);
        self.inner.cache.clear();
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

impl CacheShards {
    fn new(total_capacity: usize) -> Self {
        let shard_count = total_capacity.clamp(1, 16);
        let shard_capacity = total_capacity.div_ceil(shard_count);
        let shards = (0..shard_count)
            .map(|_| Mutex::new(QueryCache::new(shard_capacity)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { shards }
    }

    fn get(&self, key: &CacheKey) -> Option<SearchResult> {
        self.shard(key).lock().unwrap().get(key)
    }

    fn insert(&self, key: CacheKey, value: Arc<SearchResult>) {
        self.shard(&key).lock().unwrap().insert(key, value);
    }

    fn clear(&self) {
        for shard in &self.shards {
            shard.lock().unwrap().clear();
        }
    }

    fn shard(&self, key: &CacheKey) -> &Mutex<QueryCache> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        &self.shards[hasher.finish() as usize % self.shards.len()]
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
    let ordered = canonicalize_operations(operations);
    let (docs, deletes) = split_operations(ordered);
    snapshot.with_commit(generation, docs, deletes)
}

fn prepare_commit(operations: Vec<Operation>) -> Result<PreparedCommit> {
    let ordered = canonicalize_operations(operations);
    let wal_payload = encode_commit(0, &ordered)?;
    let (docs, deletes) = split_operations(ordered);
    let segment = if docs.is_empty() {
        None
    } else {
        Some(Arc::new(Segment::build(docs)))
    };
    Ok(PreparedCommit {
        wal_payload,
        segment,
        deletes,
    })
}

fn canonicalize_operations(operations: Vec<Operation>) -> Vec<Operation> {
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
    ordered
}

fn split_operations(ordered: Vec<Operation>) -> (Vec<(i64, Vec<String>)>, Vec<i64>) {
    let mut docs = Vec::new();
    let mut deletes = Vec::new();
    for operation in ordered {
        match operation {
            Operation::Upsert { rowid, fields } => docs.push((rowid, fields)),
            Operation::Delete { rowid } => deletes.push(rowid),
        }
    }
    (docs, deletes)
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

fn put_len(out: &mut Vec<u8>, len: usize, what: &'static str) -> Result<()> {
    let len = u32::try_from(len)
        .map_err(|_| Error::InvalidInput(format!("{what} count exceeds 32-bit format limit")))?;
    put_u32(out, len);
    Ok(())
}

fn encode_snapshot(snapshot: &Snapshot) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    put_u64(&mut payload, snapshot.generation);
    put_len(&mut payload, snapshot.segments.len(), "segment")?;
    for segment in &snapshot.segments {
        put_len(&mut payload, segment.docs.len(), "document")?;
        for doc in &segment.docs {
            put_i64(&mut payload, doc.rowid);
            put_u32(&mut payload, doc.len);
            put_len(&mut payload, doc.fields.len(), "field")?;
            for field in &doc.fields {
                put_bytes(&mut payload, field.as_bytes())?;
            }
        }

        put_len(&mut payload, segment.terms.len(), "term")?;
        for (term, list) in &segment.terms {
            put_bytes(&mut payload, term.as_bytes())?;
            put_len(&mut payload, list.docids.len(), "posting")?;
            for index in 0..list.docids.len() {
                put_u32(&mut payload, list.docids[index]);
                put_u32(&mut payload, list.freqs[index]);
                put_len(&mut payload, list.positions[index].len(), "position")?;
                for position in &list.positions[index] {
                    put_u32(&mut payload, *position);
                }
            }
            put_len(&mut payload, list.blocks.len(), "block")?;
            for block in &list.blocks {
                put_u32(&mut payload, block.end);
                put_u32(&mut payload, block.last_docid);
                put_u32(&mut payload, block.max_tf);
                put_u32(&mut payload, block.min_doc_len);
            }
            put_u32(&mut payload, list.max_tf);
            put_u32(&mut payload, list.min_doc_len);
        }
    }
    Ok(payload)
}

fn read_count(input: &mut &[u8], minimum_bytes: usize, message: &'static str) -> Result<usize> {
    let count = read_u32(input)? as usize;
    if minimum_bytes != 0 && count > input.len() / minimum_bytes {
        return Err(Error::Corrupt(message));
    }
    Ok(count)
}

fn decode_snapshot(mut payload: &[u8]) -> Result<Snapshot> {
    let generation = read_u64(&mut payload)?;
    let segment_count = read_count(&mut payload, 4, "invalid segment count")?;
    let mut segments = Vec::with_capacity(segment_count);

    for _ in 0..segment_count {
        let doc_count = read_count(&mut payload, 16, "invalid document count")?;
        let mut docs = Vec::with_capacity(doc_count);
        for _ in 0..doc_count {
            let rowid = read_i64(&mut payload)?;
            let len = read_u32(&mut payload)?;
            let field_count = read_count(&mut payload, 4, "invalid field count")?;
            if field_count == 0 || field_count > 64 {
                return Err(Error::Corrupt("invalid field count"));
            }
            let mut fields = Vec::with_capacity(field_count);
            for _ in 0..field_count {
                let field = read_bytes(&mut payload)?;
                fields.push(
                    String::from_utf8(field.to_vec())
                        .map_err(|_| Error::Corrupt("field is not UTF-8"))?,
                );
            }
            docs.push(StoredDocument {
                rowid,
                fields,
                len,
                unique_terms: Vec::new(),
            });
        }

        let term_count = read_count(&mut payload, 4, "invalid term count")?;
        let mut terms = std::collections::BTreeMap::new();
        for _ in 0..term_count {
            let term = String::from_utf8(read_bytes(&mut payload)?.to_vec())
                .map_err(|_| Error::Corrupt("term is not UTF-8"))?;
            if term.is_empty() || terms.contains_key(&term) {
                return Err(Error::Corrupt("invalid or duplicate term"));
            }
            let posting_count = read_count(&mut payload, 12, "invalid posting count")?;
            if posting_count == 0 || posting_count > doc_count {
                return Err(Error::Corrupt("invalid posting count"));
            }
            let mut list = PostingList {
                docids: Vec::with_capacity(posting_count),
                freqs: Vec::with_capacity(posting_count),
                positions: Vec::with_capacity(posting_count),
                blocks: Vec::new(),
                max_tf: 0,
                min_doc_len: 0,
            };
            for _ in 0..posting_count {
                let docid = read_u32(&mut payload)?;
                let frequency = read_u32(&mut payload)?;
                if docid as usize >= doc_count
                    || list
                        .docids
                        .last()
                        .is_some_and(|previous| *previous >= docid)
                {
                    return Err(Error::Corrupt("posting docids are invalid"));
                }
                let position_count = read_count(&mut payload, 4, "invalid position count")?;
                if frequency == 0 || position_count != frequency as usize {
                    return Err(Error::Corrupt("posting frequency does not match positions"));
                }
                let mut positions = Vec::with_capacity(position_count);
                for _ in 0..position_count {
                    let position = read_u32(&mut payload)?;
                    if positions
                        .last()
                        .is_some_and(|previous| *previous >= position)
                    {
                        return Err(Error::Corrupt("posting positions are not increasing"));
                    }
                    positions.push(position);
                }
                docs[docid as usize].unique_terms.push(term.clone());
                list.docids.push(docid);
                list.freqs.push(frequency);
                list.positions.push(positions);
            }

            let block_count = read_count(&mut payload, 16, "invalid block count")?;
            let expected_blocks = posting_count.div_ceil(crate::index::POSTING_BLOCK_SIZE);
            if block_count != expected_blocks {
                return Err(Error::Corrupt("invalid block count"));
            }
            list.blocks.reserve(block_count);
            for block_index in 0..block_count {
                let block = BlockMeta {
                    end: read_u32(&mut payload)?,
                    last_docid: read_u32(&mut payload)?,
                    max_tf: read_u32(&mut payload)?,
                    min_doc_len: read_u32(&mut payload)?,
                };
                let expected_end =
                    ((block_index + 1) * crate::index::POSTING_BLOCK_SIZE).min(posting_count);
                let expected_start = block_index * crate::index::POSTING_BLOCK_SIZE;
                let expected_max_tf = list.freqs[expected_start..expected_end]
                    .iter()
                    .copied()
                    .max()
                    .unwrap();
                let expected_min_doc_len = list.docids[expected_start..expected_end]
                    .iter()
                    .map(|docid| docs[*docid as usize].len)
                    .min()
                    .unwrap();
                if block.end as usize != expected_end
                    || block.last_docid != list.docids[expected_end - 1]
                    || block.max_tf != expected_max_tf
                    || block.min_doc_len != expected_min_doc_len
                {
                    return Err(Error::Corrupt("invalid block metadata"));
                }
                list.blocks.push(block);
            }
            list.max_tf = read_u32(&mut payload)?;
            list.min_doc_len = read_u32(&mut payload)?;
            let expected_max_tf = list.freqs.iter().copied().max().unwrap();
            let expected_min_doc_len = list
                .docids
                .iter()
                .map(|docid| docs[*docid as usize].len)
                .min()
                .unwrap();
            if list.max_tf != expected_max_tf || list.min_doc_len != expected_min_doc_len {
                return Err(Error::Corrupt("invalid posting-list metadata"));
            }
            terms.insert(term, list);
        }
        segments.push(Arc::new(Segment { docs, terms }));
    }
    if !payload.is_empty() {
        return Err(Error::Corrupt("trailing checkpoint bytes"));
    }

    let mut snapshot = Snapshot {
        generation,
        segments,
        latest: HashMap::new(),
        live_masks: Vec::new(),
        term_df: HashMap::new(),
        live_docs: 0,
        total_doc_len: 0,
    };
    for (segment_index, segment) in snapshot.segments.iter().enumerate() {
        let words = segment.docs.len().div_ceil(64);
        let mut live_mask = vec![u64::MAX; words];
        if let Some(last) = live_mask.last_mut()
            && segment.docs.len() % 64 != 0
        {
            *last = (1_u64 << (segment.docs.len() % 64)) - 1;
        }
        snapshot.live_masks.push(Arc::new(live_mask));
        for (docid, doc) in segment.docs.iter().enumerate() {
            if snapshot
                .latest
                .insert(
                    doc.rowid,
                    Location {
                        segment: segment_index,
                        docid: docid as u32,
                        deleted: false,
                    },
                )
                .is_some()
            {
                return Err(Error::Corrupt("duplicate rowid in checkpoint"));
            }
            snapshot.live_docs += 1;
            snapshot.total_doc_len += u64::from(doc.len);
            for term in &doc.unique_terms {
                *snapshot.term_df.entry(term.clone()).or_default() += 1;
            }
        }
    }
    Ok(snapshot)
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
        if len > MAX_WAL_RECORD_BYTES {
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
        let Ok(snapshot) = read_checkpoint(&candidate) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| snapshot.generation > current.generation)
        {
            best = Some(snapshot);
        }
    }
    Ok(best)
}

fn remove_old_checkpoints(directory: &Path, keep: &Path) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path == keep {
            continue;
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with("segment-") && file_name.ends_with(".kst") {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn read_checkpoint(path: &Path) -> Result<Snapshot> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 16 || (&bytes[..8] != SEGMENT_MAGIC && &bytes[..8] != LEGACY_SEGMENT_MAGIC) {
        return Err(Error::Corrupt("invalid checkpoint header"));
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let expected = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if bytes.len() != 16 + len {
        return Err(Error::Corrupt("invalid checkpoint length"));
    }
    let payload = &bytes[16..];
    if checksum(payload) != expected {
        return Err(Error::Corrupt("checkpoint checksum mismatch"));
    }
    if &bytes[..8] == LEGACY_SEGMENT_MAGIC {
        let (generation, operations) = decode_commit(payload)?;
        Ok(apply_operations(
            &Snapshot::default(),
            generation,
            operations,
        ))
    } else {
        decode_snapshot(payload)
    }
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
    fn corruption_before_wal_tail_is_rejected() {
        let path = temp_dir("corrupt");
        let db = Database::open(&path).unwrap();
        for rowid in 1..=2 {
            let mut tx = db.begin();
            tx.upsert_text(rowid, format!("commit {rowid}")).unwrap();
            tx.commit().unwrap();
        }
        drop(db);

        let wal_path = path.join("kestrel.wal");
        let mut bytes = fs::read(&wal_path).unwrap();
        bytes[20] ^= 0x80;
        fs::write(&wal_path, bytes).unwrap();
        assert!(matches!(Database::open(&path), Err(Error::Corrupt(_))));
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
            assert_eq!(fs::metadata(path.join("kestrel.wal")).unwrap().len(), 0);
            assert_eq!(
                &fs::read(
                    fs::read_dir(&path)
                        .unwrap()
                        .filter_map(std::result::Result::ok)
                        .map(|entry| entry.path())
                        .find(|path| path.extension().is_some_and(|ext| ext == "kst"))
                        .unwrap()
                )
                .unwrap()[..8],
                SEGMENT_MAGIC
            );

            let mut after_checkpoint = db.begin();
            after_checkpoint
                .upsert_text(3, "written after checkpoint")
                .unwrap();
            assert_eq!(after_checkpoint.commit().unwrap(), 3);
            db.optimize().unwrap();
            let checkpoints = fs::read_dir(&path)
                .unwrap()
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "kst"))
                .count();
            assert_eq!(checkpoints, 1);
        }
        let db = Database::open(&path).unwrap();
        assert_eq!(db.len(), 3);
        assert_eq!(db.generation(), 3);
        assert_eq!(db.get(1).unwrap(), ["new value"]);
        assert_eq!(db.get(3).unwrap(), ["written after checkpoint"]);

        let mut tx = db.begin();
        tx.upsert_text(4, "new WAL generation").unwrap();
        assert_eq!(tx.commit().unwrap(), 4);
        drop(db);
        let reopened = Database::open(&path).unwrap();
        assert_eq!(reopened.generation(), 4);
        assert_eq!(reopened.len(), 4);
        drop(reopened);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn native_checkpoint_restores_query_ready_postings() {
        let path = temp_dir("native-checkpoint");
        fs::create_dir_all(&path).unwrap();
        let mut snapshot = Snapshot::default().with_commit(
            9,
            vec![
                (1, vec!["alpha beta beta".to_owned()]),
                (2, vec!["alpha gamma".to_owned()]),
            ],
            Vec::new(),
        );
        Arc::make_mut(&mut snapshot.segments[0]).docs[0].fields[0] =
            "stored field deliberately differs".to_owned();
        let payload = encode_snapshot(&snapshot).unwrap();
        let checkpoint = path.join("segment-9-native.kst");
        write_framed_file(&checkpoint, SEGMENT_MAGIC, &payload).unwrap();

        let restored = read_checkpoint(&checkpoint).unwrap();
        let options = SearchOptions {
            cache: false,
            ..SearchOptions::default()
        };
        let result = restored.search(&Query::term("beta"), options, false);
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].rowid, 1);
        assert_eq!(result.hits[0].fields, ["stored field deliberately differs"]);
        assert!(
            restored
                .search(&Query::term("deliberately"), options, false)
                .hits
                .is_empty()
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn legacy_document_checkpoint_remains_readable() {
        let path = temp_dir("legacy-checkpoint");
        fs::create_dir_all(&path).unwrap();
        let operations = vec![Operation::Upsert {
            rowid: 41,
            fields: vec!["legacy searchable checkpoint".to_owned()],
        }];
        let payload = encode_commit(7, &operations).unwrap();
        write_framed_file(
            &path.join("segment-7-legacy.kst"),
            LEGACY_SEGMENT_MAGIC,
            &payload,
        )
        .unwrap();

        let db = Database::open(&path).unwrap();
        assert_eq!(db.generation(), 7);
        assert_eq!(db.get(41).unwrap(), ["legacy searchable checkpoint"]);
        assert_eq!(
            db.search(&Query::term("searchable"), SearchOptions::default())
                .unwrap()
                .hits[0]
                .rowid,
            41
        );
        drop(db);
        fs::remove_dir_all(path).unwrap();
    }
}
