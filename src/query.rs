use crate::tokenizer::normalize_term;

/// A typed query tree. It deliberately has no SQL or textual DSL in the hot path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Query {
    Term(String),
    Prefix(String),
    Phrase(Vec<String>),
    Near { terms: Vec<String>, distance: u32 },
    Column { column: u8, query: Box<Query> },
    And(Vec<Query>),
    Or(Vec<Query>),
    Not(Box<Query>),
    All,
}

impl Query {
    pub fn term(term: impl AsRef<str>) -> Self {
        Self::Term(normalize_term(term.as_ref()))
    }

    pub fn prefix(prefix: impl AsRef<str>) -> Self {
        Self::Prefix(normalize_term(prefix.as_ref()))
    }

    pub fn phrase<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::Phrase(
            terms
                .into_iter()
                .map(|term| normalize_term(term.as_ref()))
                .collect(),
        )
    }

    pub fn near<I, S>(terms: I, distance: u32) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::Near {
            terms: terms
                .into_iter()
                .map(|term| normalize_term(term.as_ref()))
                .collect(),
            distance,
        }
    }

    pub fn column(column: u8, query: Query) -> Self {
        Self::Column {
            column: column.min(63),
            query: Box::new(query),
        }
    }

    pub fn and(queries: impl IntoIterator<Item = Query>) -> Self {
        Self::And(queries.into_iter().collect())
    }

    pub fn or(queries: impl IntoIterator<Item = Query>) -> Self {
        Self::Or(queries.into_iter().collect())
    }

    pub fn negate(query: Query) -> Self {
        Self::Not(Box::new(query))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchOptions {
    pub limit: usize,
    pub k1: f32,
    pub b: f32,
    pub cache: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            k1: 1.2,
            b: 0.75,
            cache: true,
        }
    }
}

impl SearchOptions {
    pub(crate) fn validate(self) -> crate::Result<Self> {
        if self.limit == 0 {
            return Err(crate::Error::InvalidInput(
                "search limit must be greater than zero".to_owned(),
            ));
        }
        if !self.k1.is_finite()
            || self.k1 < 0.0
            || !self.b.is_finite()
            || !(0.0..=1.0).contains(&self.b)
        {
            return Err(crate::Error::InvalidInput(
                "BM25 requires finite k1 >= 0 and b in 0..=1".to_owned(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub rowid: i64,
    /// Positive BM25 score; larger is better.
    pub score: f32,
    pub fields: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub segments: usize,
    pub candidate_docs: usize,
    pub scored_docs: usize,
    pub posting_blocks: usize,
    pub skipped_blocks: usize,
    pub cache_hit: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub stats: SearchStats,
    pub generation: u64,
    pub is_approximate: bool,
}
