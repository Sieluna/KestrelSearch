use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
};

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
    pub docs: Vec<StoredDocument>,
    pub terms: BTreeMap<String, PostingList>,
}

impl Segment {
    pub fn build(docs: Vec<(i64, Vec<String>)>) -> Self {
        let mut segment = Self {
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

struct WandCursor<'a> {
    list: &'a PostingList,
    index: usize,
    ordinal: usize,
    idf: f32,
    scale: f32,
    list_bound: f32,
}

impl WandCursor<'_> {
    #[inline]
    fn exhausted(&self) -> bool {
        self.index >= self.list.docids.len()
    }

    #[inline]
    fn docid(&self) -> u32 {
        self.list.docids[self.index]
    }

    #[inline]
    fn block_bound(&self, avgdl: f32, options: SearchOptions) -> f32 {
        let block = &self.list.blocks[self.index / POSTING_BLOCK_SIZE];
        debug_assert!(self.docid() <= block.last_docid);
        let exact = bm25(self.idf, block.max_tf, block.min_doc_len, avgdl, options);
        dequantize_upper(quantize_upper(exact, self.scale), self.scale)
    }

    fn skip_block(&mut self) {
        self.index = self.list.blocks[self.index / POSTING_BLOCK_SIZE].end as usize;
    }

    fn advance_to(&mut self, target: u32) -> usize {
        let old_block = self.index / POSTING_BLOCK_SIZE;
        let offset = self.list.docids[self.index..].partition_point(|docid| *docid < target);
        self.index += offset;
        let new_block = self.index / POSTING_BLOCK_SIZE;
        new_block.saturating_sub(old_block)
    }
}

fn cursor_order(left: &WandCursor<'_>, right: &WandCursor<'_>) -> Ordering {
    left.docid()
        .cmp(&right.docid())
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn sort_cursors(cursors: &mut [WandCursor<'_>]) {
    cursors.sort_unstable_by(cursor_order);
}

fn restore_cursor_order(cursors: &mut Vec<WandCursor<'_>>) {
    if cursors[0].exhausted() {
        cursors.remove(0);
        return;
    }
    let mut index = 0;
    while index + 1 < cursors.len()
        && cursor_order(&cursors[index], &cursors[index + 1]) == Ordering::Greater
    {
        cursors.swap(index, index + 1);
        index += 1;
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Location {
    pub segment: usize,
    pub docid: u32,
    pub deleted: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Snapshot {
    pub generation: u64,
    pub segments: Vec<std::sync::Arc<Segment>>,
    pub latest: HashMap<i64, Location>,
    pub live_masks: Vec<std::sync::Arc<Vec<u64>>>,
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
        let segment = if docs.is_empty() {
            None
        } else {
            Some(std::sync::Arc::new(Segment::build(docs)))
        };
        self.with_prebuilt_commit(generation, segment, deletes)
    }

    pub fn with_prebuilt_commit(
        &self,
        generation: u64,
        segment: Option<std::sync::Arc<Segment>>,
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
                    segment: 0,
                    docid: 0,
                    deleted: true,
                },
            );
        }

        if let Some(segment) = segment {
            let segment_index = next.segments.len();
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
                        segment: segment_index,
                        docid: docid as u32,
                        deleted: false,
                    },
                );
            }
            next.live_masks
                .push(std::sync::Arc::new(full_live_mask(segment.docs.len())));
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
        let mask = std::sync::Arc::make_mut(&mut self.live_masks[location.segment]);
        mask[location.docid as usize / 64] &= !(1_u64 << (location.docid % 64));
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
        self.live_masks[segment][docid as usize / 64] & (1_u64 << (docid % 64)) != 0
    }

    pub fn search(&self, query: &Query, options: SearchOptions, exhaustive: bool) -> SearchResult {
        let mut stats = SearchStats {
            segments: self.segments.len(),
            ..SearchStats::default()
        };
        let terms = self.expanded_scoring_terms(query);
        let scoring_terms: Vec<_> = terms
            .iter()
            .map(|term| ScoringTerm {
                term,
                idf: self.idf(term),
            })
            .collect();
        let direct_disjunction = is_flat_disjunction(query);
        let direct_conjunction = is_flat_conjunction(query);
        let phrase_order = match query {
            Query::Phrase(phrase) if phrase.len() == scoring_terms.len() => Some(
                phrase
                    .iter()
                    .map(|term| terms.binary_search(term).unwrap())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };
        let avgdl = if self.live_docs == 0 {
            1.0
        } else {
            self.total_doc_len as f32 / self.live_docs as f32
        };
        let mut ranked = TopK::new(options.limit);

        if exhaustive || terms.is_empty() {
            for location in self.latest.values().filter(|location| !location.deleted) {
                stats.candidate_docs += 1;
                self.consider(
                    query,
                    &scoring_terms,
                    location.segment,
                    location.docid,
                    avgdl,
                    options,
                    &mut ranked,
                    &mut stats,
                );
            }
        } else {
            for (segment_index, segment) in self.segments.iter().enumerate() {
                let mut cursors: Vec<_> = scoring_terms
                    .iter()
                    .enumerate()
                    .filter_map(|(ordinal, scoring)| {
                        let list = segment.terms.get(scoring.term)?;
                        let exact_bound =
                            bm25(scoring.idf, list.max_tf, list.min_doc_len, avgdl, options);
                        let scale = exact_bound / f32::from(u16::MAX);
                        let list_bound =
                            dequantize_upper(quantize_upper(exact_bound, scale), scale);
                        Some(WandCursor {
                            list,
                            index: 0,
                            ordinal,
                            idf: scoring.idf,
                            scale,
                            list_bound,
                        })
                    })
                    .collect();
                if direct_disjunction && scoring_terms.len() == 1 && cursors.len() == 1 {
                    self.single_list_segment(
                        segment_index,
                        &cursors[0],
                        avgdl,
                        options,
                        &mut ranked,
                        &mut stats,
                    );
                    continue;
                }
                if direct_conjunction || phrase_order.is_some() {
                    if cursors.len() == scoring_terms.len() {
                        self.intersect_segment(
                            segment_index,
                            &mut cursors,
                            phrase_order.as_deref(),
                            avgdl,
                            options,
                            &mut ranked,
                            &mut stats,
                        );
                    }
                    continue;
                }
                sort_cursors(&mut cursors);

                while !cursors.is_empty() {
                    let theta = ranked.threshold();

                    // A first-list block can be discarded only after adding every
                    // other cursor's list-wide upper bound.
                    if ranked.is_full() && (!direct_disjunction || cursors.len() == 1) {
                        stats.posting_blocks += 1;
                        let first_block_bound = f64::from(cursors[0].block_bound(avgdl, options))
                            + cursors[1..]
                                .iter()
                                .map(|cursor| f64::from(cursor.list_bound))
                                .sum::<f64>();
                        if first_block_bound < f64::from(theta) {
                            cursors[0].skip_block();
                            restore_cursor_order(&mut cursors);
                            stats.skipped_blocks += 1;
                            continue;
                        }
                    }

                    let mut accumulated = 0.0_f64;
                    let pivot = cursors.iter().position(|cursor| {
                        accumulated += f64::from(cursor.list_bound);
                        !ranked.is_full() || accumulated >= f64::from(theta)
                    });
                    let Some(pivot) = pivot else {
                        break;
                    };
                    let pivot_doc = cursors[pivot].docid();

                    if cursors[0].docid() < pivot_doc {
                        stats.skipped_blocks += cursors[0].advance_to(pivot_doc);
                        restore_cursor_order(&mut cursors);
                        continue;
                    }

                    stats.candidate_docs += 1;
                    if direct_disjunction {
                        self.consider_wand_candidate(
                            segment_index,
                            pivot_doc,
                            &cursors,
                            avgdl,
                            options,
                            &mut ranked,
                            &mut stats,
                        );
                    } else {
                        self.consider(
                            query,
                            &scoring_terms,
                            segment_index,
                            pivot_doc,
                            avgdl,
                            options,
                            &mut ranked,
                            &mut stats,
                        );
                    }
                    let matching = cursors
                        .iter()
                        .take_while(|cursor| cursor.docid() == pivot_doc)
                        .count();
                    for cursor in &mut cursors[..matching] {
                        cursor.index += 1;
                    }
                    cursors.retain(|cursor| !cursor.exhausted());
                    sort_cursors(&mut cursors);
                }
            }
        }

        let hits = ranked
            .into_sorted()
            .into_iter()
            .map(|ranked| {
                let doc = &self.segments[ranked.segment].docs[ranked.docid as usize];
                SearchHit {
                    rowid: ranked.rowid,
                    score: ranked.score,
                    fields: doc.fields.clone(),
                }
            })
            .collect();
        SearchResult {
            hits,
            stats,
            generation: self.generation,
            is_approximate: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn consider(
        &self,
        query: &Query,
        terms: &[ScoringTerm<'_>],
        segment_index: usize,
        docid: u32,
        avgdl: f32,
        options: SearchOptions,
        ranked: &mut TopK,
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
        let budget = bm25_budget(doc.len, avgdl, options);
        let score = terms
            .iter()
            .filter_map(|scoring| {
                let list = segment.terms.get(scoring.term)?;
                let index = list.find(docid)?;
                Some(bm25_with_budget(
                    scoring.idf,
                    list.freqs[index],
                    budget,
                    options.k1,
                ))
            })
            .sum();
        let hit = RankedDoc {
            segment: segment_index,
            docid,
            rowid: doc.rowid,
            score,
        };
        ranked.push(hit);
    }

    #[allow(clippy::too_many_arguments)]
    fn consider_wand_candidate(
        &self,
        segment_index: usize,
        docid: u32,
        cursors: &[WandCursor<'_>],
        avgdl: f32,
        options: SearchOptions,
        ranked: &mut TopK,
        stats: &mut SearchStats,
    ) {
        if !self.is_live(segment_index, docid) {
            return;
        }
        stats.scored_docs += 1;
        let doc = &self.segments[segment_index].docs[docid as usize];
        let budget = bm25_budget(doc.len, avgdl, options);
        let score = cursors
            .iter()
            .take_while(|cursor| cursor.docid() == docid)
            .map(|cursor| {
                bm25_with_budget(
                    cursor.idf,
                    cursor.list.freqs[cursor.index],
                    budget,
                    options.k1,
                )
            })
            .sum();
        ranked.push(RankedDoc {
            segment: segment_index,
            docid,
            rowid: doc.rowid,
            score,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn intersect_segment(
        &self,
        segment_index: usize,
        cursors: &mut [WandCursor<'_>],
        phrase_order: Option<&[usize]>,
        avgdl: f32,
        options: SearchOptions,
        ranked: &mut TopK,
        stats: &mut SearchStats,
    ) {
        while cursors.iter().all(|cursor| !cursor.exhausted()) {
            let mut target = cursors.iter().map(WandCursor::docid).max().unwrap();
            let mut aligned = false;
            while !aligned {
                aligned = true;
                for cursor in cursors.iter_mut() {
                    if cursor.docid() < target {
                        stats.skipped_blocks += cursor.advance_to(target);
                        if cursor.exhausted() {
                            return;
                        }
                    }
                    if cursor.docid() > target {
                        target = cursor.docid();
                        aligned = false;
                    }
                }
            }

            stats.candidate_docs += 1;
            if self.is_live(segment_index, target)
                && phrase_order.is_none_or(|order| cursor_phrase_matches(cursors, order))
            {
                stats.scored_docs += 1;
                let doc = &self.segments[segment_index].docs[target as usize];
                let budget = bm25_budget(doc.len, avgdl, options);
                let score = cursors
                    .iter()
                    .map(|cursor| {
                        bm25_with_budget(
                            cursor.idf,
                            cursor.list.freqs[cursor.index],
                            budget,
                            options.k1,
                        )
                    })
                    .sum();
                ranked.push(RankedDoc {
                    segment: segment_index,
                    docid: target,
                    rowid: doc.rowid,
                    score,
                });
            }
            for cursor in cursors.iter_mut() {
                cursor.index += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn single_list_segment(
        &self,
        segment_index: usize,
        cursor: &WandCursor<'_>,
        avgdl: f32,
        options: SearchOptions,
        ranked: &mut TopK,
        stats: &mut SearchStats,
    ) {
        for (block_index, block) in cursor.list.blocks.iter().enumerate() {
            stats.posting_blocks += 1;
            let exact_bound = bm25(cursor.idf, block.max_tf, block.min_doc_len, avgdl, options);
            let block_bound =
                dequantize_upper(quantize_upper(exact_bound, cursor.scale), cursor.scale);
            if ranked.is_full() && block_bound < ranked.threshold() {
                stats.skipped_blocks += 1;
                continue;
            }
            let start = block_index * POSTING_BLOCK_SIZE;
            for index in start..block.end as usize {
                let docid = cursor.list.docids[index];
                stats.candidate_docs += 1;
                if !self.is_live(segment_index, docid) {
                    continue;
                }
                stats.scored_docs += 1;
                let doc = &self.segments[segment_index].docs[docid as usize];
                ranked.push(RankedDoc {
                    segment: segment_index,
                    docid,
                    rowid: doc.rowid,
                    score: bm25(
                        cursor.idf,
                        cursor.list.freqs[index],
                        doc.len,
                        avgdl,
                        options,
                    ),
                });
            }
        }
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
            Query::Phrase(phrase) | Query::Near { terms: phrase, .. } if positive => {
                terms.extend(phrase.iter().cloned());
            }
            Query::Column { query, .. } => self.collect_terms(query, positive, terms),
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
    matches_query_in_column(segment, docid, query, None)
}

fn is_flat_disjunction(query: &Query) -> bool {
    match query {
        Query::Term(_) | Query::Prefix(_) => true,
        Query::Or(children) => {
            !children.is_empty()
                && children
                    .iter()
                    .all(|child| matches!(child, Query::Term(_) | Query::Prefix(_)))
        }
        _ => false,
    }
}

fn is_flat_conjunction(query: &Query) -> bool {
    matches!(query, Query::And(children)
        if !children.is_empty() && children.iter().all(|child| matches!(child, Query::Term(_))))
}

fn cursor_phrase_matches(cursors: &[WandCursor<'_>], phrase_order: &[usize]) -> bool {
    let first_cursor = &cursors[phrase_order[0]];
    let first = &first_cursor.list.positions[first_cursor.index];
    first.iter().any(|start| {
        let column = start >> COLUMN_SHIFT;
        phrase_order[1..]
            .iter()
            .enumerate()
            .all(|(offset, ordinal)| {
                let cursor = &cursors[*ordinal];
                let positions = &cursor.list.positions[cursor.index];
                let target = start.saturating_add(offset as u32 + 1);
                target >> COLUMN_SHIFT == column && positions.binary_search(&target).is_ok()
            })
    })
}

fn matches_query_in_column(
    segment: &Segment,
    docid: u32,
    query: &Query,
    column: Option<u8>,
) -> bool {
    match query {
        Query::Term(term) => term_positions(segment, docid, term)
            .is_some_and(|positions| positions_in_column(positions, column)),
        Query::Prefix(prefix) => segment
            .terms
            .range(prefix.clone()..)
            .take_while(|(term, _)| term.starts_with(prefix))
            .any(|(_, list)| {
                list.find(docid)
                    .is_some_and(|index| positions_in_column(&list.positions[index], column))
            }),
        Query::Phrase(terms) => matches_phrase(segment, docid, terms, column),
        Query::Near { terms, distance } => matches_near(segment, docid, terms, *distance, column),
        Query::Column {
            column: restricted,
            query,
        } => {
            if column.is_some_and(|outer| outer != *restricted) {
                false
            } else {
                matches_query_in_column(segment, docid, query, Some(*restricted))
            }
        }
        Query::And(children) => children
            .iter()
            .all(|child| matches_query_in_column(segment, docid, child, column)),
        Query::Or(children) => children
            .iter()
            .any(|child| matches_query_in_column(segment, docid, child, column)),
        Query::Not(child) => !matches_query_in_column(segment, docid, child, column),
        Query::All => true,
    }
}

fn matches_phrase(
    segment: &Segment,
    docid: u32,
    terms: &[String],
    required_column: Option<u8>,
) -> bool {
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
        if required_column.is_some_and(|required| u32::from(required) != column) {
            return false;
        }
        rest.iter().enumerate().all(|(offset, positions)| {
            let target = start.saturating_add(offset as u32 + 1);
            target >> COLUMN_SHIFT == column && positions.binary_search(&target).is_ok()
        })
    })
}

fn matches_near(
    segment: &Segment,
    docid: u32,
    terms: &[String],
    distance: u32,
    required_column: Option<u8>,
) -> bool {
    if terms.is_empty() {
        return false;
    }
    let mut requirements = Vec::<(&str, usize)>::new();
    for term in terms {
        if let Some((_, count)) = requirements
            .iter_mut()
            .find(|(existing, _)| *existing == term)
        {
            *count += 1;
        } else {
            requirements.push((term, 1));
        }
    }
    let mut events = Vec::<(u32, usize)>::new();
    for (term_index, (term, _)) in requirements.iter().enumerate() {
        let Some(positions) = term_positions(segment, docid, term) else {
            return false;
        };
        for &position in positions {
            let column = (position >> COLUMN_SHIFT) as u8;
            if required_column.is_none_or(|required| required == column) {
                events.push((position, term_index));
            }
        }
    }
    events.sort_unstable();

    let mut counts = vec![0_usize; requirements.len()];
    let mut covered = 0_usize;
    let mut left = 0_usize;
    for right in 0..events.len() {
        let (_, term) = events[right];
        counts[term] += 1;
        if counts[term] == requirements[term].1 {
            covered += 1;
        }

        while covered == requirements.len() {
            let (left_position, left_term) = events[left];
            let (right_position, _) = events[right];
            let same_column = left_position >> COLUMN_SHIFT == right_position >> COLUMN_SHIFT;
            let span = (right_position & ((1 << COLUMN_SHIFT) - 1))
                .saturating_sub(left_position & ((1 << COLUMN_SHIFT) - 1));
            if same_column && span <= distance {
                return true;
            }
            if counts[left_term] == requirements[left_term].1 {
                covered -= 1;
            }
            counts[left_term] -= 1;
            left += 1;
        }
    }
    false
}

fn term_positions<'a>(segment: &'a Segment, docid: u32, term: &str) -> Option<&'a [u32]> {
    let list = segment.terms.get(term)?;
    let index = list.find(docid)?;
    Some(&list.positions[index])
}

fn positions_in_column(positions: &[u32], column: Option<u8>) -> bool {
    column.is_none_or(|column| {
        positions
            .iter()
            .any(|position| position >> COLUMN_SHIFT == u32::from(column))
    })
}

#[inline]
fn bm25(idf: f32, freq: u32, doc_len: u32, avgdl: f32, options: SearchOptions) -> f32 {
    if freq == 0 {
        return 0.0;
    }
    bm25_with_budget(idf, freq, bm25_budget(doc_len, avgdl, options), options.k1)
}

#[inline(always)]
fn bm25_budget(doc_len: u32, avgdl: f32, options: SearchOptions) -> f32 {
    options.k1 * (1.0 - options.b + options.b * doc_len as f32 / avgdl.max(1.0))
}

#[inline(always)]
fn bm25_with_budget(idf: f32, freq: u32, budget: f32, k1: f32) -> f32 {
    let f = freq as f32;
    idf * (f * (k1 + 1.0)) / (f + budget)
}

#[derive(Clone, Copy)]
struct ScoringTerm<'a> {
    term: &'a str,
    idf: f32,
}

#[derive(Clone, Copy, Debug)]
struct RankedDoc {
    segment: usize,
    docid: u32,
    rowid: i64,
    score: f32,
}

impl PartialEq for RankedDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.rowid == other.rowid
            && self.segment == other.segment
            && self.docid == other.docid
    }
}

impl Eq for RankedDoc {}

impl PartialOrd for RankedDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.rowid.cmp(&other.rowid))
            .then_with(|| self.segment.cmp(&other.segment))
            .then_with(|| self.docid.cmp(&other.docid))
    }
}

struct TopK {
    limit: usize,
    heap: BinaryHeap<RankedDoc>,
}

impl TopK {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit),
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.heap.len() == self.limit
    }

    #[inline]
    fn threshold(&self) -> f32 {
        if self.is_full() {
            self.heap.peek().unwrap().score
        } else {
            0.0
        }
    }

    #[inline]
    fn push(&mut self, hit: RankedDoc) {
        if !self.is_full() {
            self.heap.push(hit);
        } else if hit < *self.heap.peek().unwrap() {
            *self.heap.peek_mut().unwrap() = hit;
        }
    }

    fn into_sorted(self) -> Vec<RankedDoc> {
        let mut ranked = self.heap.into_vec();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.rowid.cmp(&right.rowid))
        });
        ranked
    }
}

fn full_live_mask(docs: usize) -> Vec<u64> {
    let mut mask = vec![u64::MAX; docs.div_ceil(64)];
    if let Some(last) = mask.last_mut() {
        let remainder = docs % 64;
        if remainder != 0 {
            *last = (1_u64 << remainder) - 1;
        }
    }
    mask
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
        let segment = Segment::build(docs);
        let list = &segment.terms["same"];
        assert_eq!(list.blocks.len(), 3);
        assert_eq!(list.blocks[0].end, 128);
        assert_eq!(list.blocks[2].end, 300);
    }
}
