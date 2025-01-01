use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    query::{Query, SearchHit, SearchOptions, SearchResult, SearchStats},
    tokenizer::{COLUMN_SHIFT, group_positions, tokenize_fields},
};

pub(crate) const POSTING_BLOCK_SIZE: usize = 128;

#[derive(Clone, Debug)]
pub(crate) struct StoredDocument {
    pub rowid: i64,
    pub fields: Vec<String>,
    pub len: u32,
    pub unique_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockMeta {
    pub start: u32,
    pub end: u32,
    pub last_docid: u32,
    pub max_tf: u32,
    pub min_doc_len: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PostingList {
    /// Hot matching stream.
    pub docids: Vec<u32>,
    /// Hot scoring stream, physically separate from positions.
    pub freqs: Vec<u32>,
    /// Cold phrase/snippet stream; only accessed during positional verification.
    pub positions: Vec<Vec<u32>>,
    pub blocks: Vec<BlockMeta>,
    pub max_tf: u32,
    pub min_doc_len: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Segment {
    pub generation: u64,
    pub docs: Vec<StoredDocument>,
    pub terms: BTreeMap<String, PostingList>,
}

impl Segment {
    pub fn build(generation: u64, docs: Vec<(i64, Vec<String>)>) -> Self {
        let mut segment = Self {
            generation,
            docs: Vec::with_capacity(docs.len()),
            terms: BTreeMap::new(),
        };

        for (docid, (rowid, fields)) in docs.into_iter().enumerate() {
            let (tokens, len) = tokenize_fields(&fields);
            let grouped = group_positions(tokens);
            let unique_terms = grouped.keys().cloned().collect();
            segment.docs.push(StoredDocument {
                rowid,
                fields,
                len,
                unique_terms,
            });

            for (term, positions) in grouped {
                let postings = segment.terms.entry(term).or_default();
                postings.docids.push(docid as u32);
                postings.freqs.push(positions.len() as u32);
                postings.positions.push(positions);
            }
        }

        for postings in segment.terms.values_mut() {
            postings.finish_blocks(&segment.docs);
        }
        segment
    }
}

impl PostingList {
    fn finish_blocks(&mut self, docs: &[StoredDocument]) {
        self.max_tf = self.freqs.iter().copied().max().unwrap_or(0);
        self.min_doc_len = self
            .docids
            .iter()
            .map(|docid| docs[*docid as usize].len)
            .min()
            .unwrap_or(0);

        for start in (0..self.docids.len()).step_by(POSTING_BLOCK_SIZE) {
            let end = (start + POSTING_BLOCK_SIZE).min(self.docids.len());
            let docids = &self.docids[start..end];
            let freqs = &self.freqs[start..end];
            self.blocks.push(BlockMeta {
                start: start as u32,
                end: end as u32,
                last_docid: *docids.last().unwrap(),
                max_tf: freqs.iter().copied().max().unwrap(),
                min_doc_len: docids
                    .iter()
                    .map(|docid| docs[*docid as usize].len)
                    .min()
                    .unwrap(),
            });
        }
    }

    #[inline]
    pub fn find(&self, docid: u32) -> Option<usize> {
        self.docids.binary_search(&docid).ok()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Location {
    pub generation: u64,
    pub segment: usize,
    pub docid: u32,
    pub deleted: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Snapshot {
    pub generation: u64,
    pub segments: Vec<std::sync::Arc<Segment>>,
    pub latest: HashMap<i64, Location>,
    pub term_df: HashMap<String, u32>,
    pub live_docs: u64,
    pub total_doc_len: u64,
}

impl Snapshot {
    pub fn with_commit(
        &self,
        generation: u64,
        docs: Vec<(i64, Vec<String>)>,
        deletes: Vec<i64>,
    ) -> Self {
        let mut next = self.clone();
        next.generation = generation;
        next.latest = self.latest.clone();
        next.term_df = self.term_df.clone();

        for rowid in &deletes {
            next.remove_latest(*rowid);
            next.latest.insert(
                *rowid,
                Location {
                    generation,
                    segment: 0,
                    docid: 0,
                    deleted: true,
                },
            );
        }

        if !docs.is_empty() {
            let segment_index = next.segments.len();
            let segment = std::sync::Arc::new(Segment::build(generation, docs));
            for (docid, doc) in segment.docs.iter().enumerate() {
                next.remove_latest(doc.rowid);
                next.live_docs += 1;
                next.total_doc_len += u64::from(doc.len);
                for term in &doc.unique_terms {
                    *next.term_df.entry(term.clone()).or_default() += 1;
                }
                next.latest.insert(
                    doc.rowid,
                    Location {
                        generation,
                        segment: segment_index,
                        docid: docid as u32,
                        deleted: false,
                    },
                );
            }
            next.segments.push(segment);
        }
        next
    }

    fn remove_latest(&mut self, rowid: i64) {
        let Some(location) = self.latest.get(&rowid).copied() else {
            return;
        };
        if location.deleted {
            return;
        }
        let doc = &self.segments[location.segment].docs[location.docid as usize];
        self.live_docs -= 1;
        self.total_doc_len -= u64::from(doc.len);
        for term in &doc.unique_terms {
            if let Some(df) = self.term_df.get_mut(term) {
                *df -= 1;
                if *df == 0 {
                    self.term_df.remove(term);
                }
            }
        }
    }

    pub fn compacted(&self) -> Self {
        let mut live: Vec<_> = self
            .latest
            .values()
            .filter(|location| !location.deleted)
            .map(|location| {
                let doc = &self.segments[location.segment].docs[location.docid as usize];
                (doc.rowid, doc.fields.clone())
            })
            .collect();
        live.sort_by_key(|(rowid, _)| *rowid);
        Snapshot::default().with_commit(self.generation, live, Vec::new())
    }

    #[inline]
    fn is_live(&self, segment: usize, docid: u32) -> bool {
        let doc = &self.segments[segment].docs[docid as usize];
        matches!(self.latest.get(&doc.rowid), Some(location)
            if !location.deleted
                && location.segment == segment
                && location.docid == docid
                && location.generation == self.segments[segment].generation)
    }

    pub fn search(&self, query: &Query, options: SearchOptions, exhaustive: bool) -> SearchResult {
        let mut stats = SearchStats {
            segments: self.segments.len(),
            ..SearchStats::default()
        };
        let terms = self.expanded_scoring_terms(query);
        let avgdl = if self.live_docs == 0 {
            1.0
        } else {
            self.total_doc_len as f32 / self.live_docs as f32
        };
        let global_max: HashMap<&str, f32> = terms
            .iter()
            .map(|term| {
                let idf = self.idf(term);
                let max = self
                    .segments
                    .iter()
                    .filter_map(|segment| segment.terms.get(term))
                    .map(|list| bm25(idf, list.max_tf, list.min_doc_len, avgdl, options))
                    .fold(0.0, f32::max);
                (term.as_str(), max)
            })
            .collect();
        let score_scales: HashMap<&str, f32> = global_max
            .iter()
            .map(|(term, max)| (*term, *max / f32::from(u16::MAX)))
            .collect();
        let mut ranked = Vec::<SearchHit>::with_capacity(options.limit);
        let mut seen = HashSet::<(usize, u32)>::new();

        if terms.is_empty() {
            for location in self.latest.values().filter(|location| !location.deleted) {
                self.consider(
                    query,
                    &terms,
                    location.segment,
                    location.docid,
                    avgdl,
                    options,
                    &mut ranked,
                    &mut stats,
                );
            }
        } else {
            let mut ordered_terms = terms.clone();
            ordered_terms.sort_by(|a, b| {
                global_max[b.as_str()]
                    .total_cmp(&global_max[a.as_str()])
                    .then_with(|| a.cmp(b))
            });

            for term in &ordered_terms {
                let term_other_max: f64 = ordered_terms
                    .iter()
                    .filter(|other| *other != term)
                    .map(|other| f64::from(global_max[other.as_str()]))
                    .sum();
                let idf = self.idf(term);
                for (segment_index, segment) in self.segments.iter().enumerate() {
                    let Some(list) = segment.terms.get(term) else {
                        continue;
                    };
                    for block in &list.blocks {
                        stats.posting_blocks += 1;
                        let theta = threshold(&ranked, options.limit);
                        let exact_block_bound =
                            bm25(idf, block.max_tf, block.min_doc_len, avgdl, options);
                        let block_bound = f64::from(dequantize_upper(
                            quantize_upper(exact_block_bound, score_scales[term.as_str()]),
                            score_scales[term.as_str()],
                        )) + term_other_max;
                        // Strict comparison preserves the deterministic rowid tie-break.
                        if !exhaustive
                            && ranked.len() == options.limit
                            && block_bound < f64::from(theta)
                        {
                            stats.skipped_blocks += 1;
                            continue;
                        }
                        debug_assert_eq!(list.docids[block.end as usize - 1], block.last_docid);
                        for &docid in &list.docids[block.start as usize..block.end as usize] {
                            if seen.insert((segment_index, docid)) {
                                stats.candidate_docs += 1;
                                self.consider(
                                    query,
                                    &terms,
                                    segment_index,
                                    docid,
                                    avgdl,
                                    options,
                                    &mut ranked,
                                    &mut stats,
                                );
                            }
                        }
                    }
                }
            }
        }

        ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.rowid.cmp(&b.rowid))
        });
        ranked.truncate(options.limit);
        SearchResult {
            hits: ranked,
            stats,
            generation: self.generation,
            is_approximate: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn consider(
        &self,
        query: &Query,
        terms: &[String],
        segment_index: usize,
        docid: u32,
        avgdl: f32,
        options: SearchOptions,
        ranked: &mut Vec<SearchHit>,
        stats: &mut SearchStats,
    ) {
        if !self.is_live(segment_index, docid) {
            return;
        }
        let segment = &self.segments[segment_index];
        if !matches_query(segment, docid, query) {
            return;
        }
        stats.scored_docs += 1;
        let doc = &segment.docs[docid as usize];
        let score = terms
            .iter()
            .filter_map(|term| {
                let list = segment.terms.get(term)?;
                let index = list.find(docid)?;
                Some(bm25(
                    self.idf(term),
                    list.freqs[index],
                    doc.len,
                    avgdl,
                    options,
                ))
            })
            .sum();
        let hit = SearchHit {
            rowid: doc.rowid,
            score,
            fields: doc.fields.clone(),
        };
        push_top_k(ranked, hit, options.limit);
    }

    fn idf(&self, term: &str) -> f32 {
        let n = self.live_docs as f32;
        let df = self.term_df.get(term).copied().unwrap_or(0) as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    fn expanded_scoring_terms(&self, query: &Query) -> Vec<String> {
        let mut terms = HashSet::new();
        self.collect_terms(query, true, &mut terms);
        let mut terms: Vec<_> = terms.into_iter().collect();
        terms.sort();
        terms
    }

    fn collect_terms(&self, query: &Query, positive: bool, terms: &mut HashSet<String>) {
        match query {
            Query::Term(term) if positive => {
                terms.insert(term.clone());
            }
            Query::Prefix(prefix) if positive => {
                for term in self.term_df.keys().filter(|term| term.starts_with(prefix)) {
                    terms.insert(term.clone());
                }
            }
            Query::Phrase(phrase) if positive => terms.extend(phrase.iter().cloned()),
            Query::And(children) | Query::Or(children) => {
                for child in children {
                    self.collect_terms(child, positive, terms);
                }
            }
            Query::Not(child) => self.collect_terms(child, false, terms),
            _ => {}
        }
    }
}

fn matches_query(segment: &Segment, docid: u32, query: &Query) -> bool {
    match query {
        Query::Term(term) => segment
            .terms
            .get(term)
            .is_some_and(|list| list.find(docid).is_some()),
        Query::Prefix(prefix) => segment
            .terms
            .range(prefix.clone()..)
            .take_while(|(term, _)| term.starts_with(prefix))
            .any(|(_, list)| list.find(docid).is_some()),
        Query::Phrase(terms) => matches_phrase(segment, docid, terms),
        Query::And(children) => children
            .iter()
            .all(|child| matches_query(segment, docid, child)),
        Query::Or(children) => children
            .iter()
            .any(|child| matches_query(segment, docid, child)),
        Query::Not(child) => !matches_query(segment, docid, child),
        Query::All => true,
    }
}

fn matches_phrase(segment: &Segment, docid: u32, terms: &[String]) -> bool {
    if terms.is_empty() {
        return false;
    }
    let Some(first) = segment
        .terms
        .get(&terms[0])
        .and_then(|list| list.find(docid).map(|i| &list.positions[i]))
    else {
        return false;
    };
    let rest: Option<Vec<&[u32]>> = terms[1..]
        .iter()
        .map(|term| {
            let list = segment.terms.get(term)?;
            let index = list.find(docid)?;
            Some(list.positions[index].as_slice())
        })
        .collect();
    let Some(rest) = rest else { return false };
    first.iter().any(|start| {
        let column = start >> COLUMN_SHIFT;
        rest.iter().enumerate().all(|(offset, positions)| {
            let target = start.saturating_add(offset as u32 + 1);
            target >> COLUMN_SHIFT == column && positions.binary_search(&target).is_ok()
        })
    })
}

#[inline]
fn bm25(idf: f32, freq: u32, doc_len: u32, avgdl: f32, options: SearchOptions) -> f32 {
    if freq == 0 {
        return 0.0;
    }
    let f = freq as f32;
    let budget = options.k1 * (1.0 - options.b + options.b * doc_len as f32 / avgdl.max(1.0));
    idf * (f * (options.k1 + 1.0)) / (f + budget)
}

fn push_top_k(ranked: &mut Vec<SearchHit>, hit: SearchHit, k: usize) {
    if ranked.len() < k {
        ranked.push(hit);
    } else if let Some((worst, _)) = ranked.iter().enumerate().min_by(|(_, a), (_, b)| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| b.rowid.cmp(&a.rowid))
    }) && (hit.score > ranked[worst].score
        || (hit.score == ranked[worst].score && hit.rowid < ranked[worst].rowid))
    {
        ranked[worst] = hit;
    }
}

fn threshold(ranked: &[SearchHit], k: usize) -> f32 {
    if ranked.len() < k {
        0.0
    } else {
        ranked
            .iter()
            .map(|hit| hit.score)
            .fold(f32::INFINITY, f32::min)
    }
}

pub(crate) fn quantize_upper(score: f32, scale: f32) -> u16 {
    if score <= 0.0 || scale <= 0.0 {
        return 0;
    }
    (score / scale).ceil().min(u16::MAX as f32) as u16
}

pub(crate) fn dequantize_upper(value: u16, scale: f32) -> f32 {
    let decoded = f32::from(value) * scale;
    if decoded > 0.0 && decoded.is_finite() {
        f32::from_bits(decoded.to_bits() + 1)
    } else {
        decoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_never_rounds_down() {
        for i in 1..100_000_u32 {
            let score = i as f32 * 0.000_31;
            let scale = 37.0 / 65_535.0;
            let decoded = dequantize_upper(quantize_upper(score, scale), scale);
            assert!(decoded + f32::EPSILON >= score, "{decoded} < {score}");
        }
    }

    #[test]
    fn block_layout_is_complete() {
        let docs = (0..300)
            .map(|id| (id, vec!["same term".to_owned()]))
            .collect();
        let segment = Segment::build(1, docs);
        let list = &segment.terms["same"];
        assert_eq!(list.blocks.len(), 3);
        assert_eq!(list.blocks[0].end, 128);
        assert_eq!(list.blocks[2].end, 300);
    }
}
