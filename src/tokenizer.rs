use std::collections::BTreeMap;

pub const COLUMN_SHIFT: u32 = 26;
pub const MAX_POSITION: u32 = (1 << COLUMN_SHIFT) - 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TokenOccurrence {
    pub term: String,
    pub position: u32,
}

/// Dependency-free Unicode tokenizer. Latin/digit runs become lowercase terms;
/// CJK runs become overlapping bigrams so Chinese, Japanese and Korean text is
/// searchable without a runtime dictionary.
pub(crate) fn tokenize_fields(fields: &[String]) -> (Vec<TokenOccurrence>, u32) {
    let mut tokens = Vec::new();
    let mut total = 0_u32;

    for (column, field) in fields.iter().enumerate() {
        let column = (column as u32).min(63);
        let mut word = String::new();
        let mut cjk = Vec::new();
        let mut offset = 0_u32;

        let flush_word =
            |word: &mut String, tokens: &mut Vec<TokenOccurrence>, offset: &mut u32| {
                if !word.is_empty() {
                    tokens.push(TokenOccurrence {
                        term: word.to_lowercase(),
                        position: (column << COLUMN_SHIFT) | (*offset).min(MAX_POSITION),
                    });
                    *offset = offset.saturating_add(1);
                    word.clear();
                }
            };
        let flush_cjk =
            |cjk: &mut Vec<char>, tokens: &mut Vec<TokenOccurrence>, offset: &mut u32| {
                if cjk.is_empty() {
                    return;
                }
                if cjk.len() == 1 {
                    tokens.push(TokenOccurrence {
                        term: cjk[0].to_string(),
                        position: (column << COLUMN_SHIFT) | (*offset).min(MAX_POSITION),
                    });
                    *offset = offset.saturating_add(1);
                } else {
                    for pair in cjk.windows(2) {
                        let term = pair.iter().collect();
                        tokens.push(TokenOccurrence {
                            term,
                            position: (column << COLUMN_SHIFT) | (*offset).min(MAX_POSITION),
                        });
                        *offset = offset.saturating_add(1);
                    }
                }
                cjk.clear();
            };

        for ch in field.chars() {
            if is_cjk(ch) {
                flush_word(&mut word, &mut tokens, &mut offset);
                cjk.push(ch);
            } else {
                flush_cjk(&mut cjk, &mut tokens, &mut offset);
                if ch.is_alphanumeric() || ch == '_' {
                    word.push(ch);
                } else {
                    flush_word(&mut word, &mut tokens, &mut offset);
                }
            }
        }
        flush_word(&mut word, &mut tokens, &mut offset);
        flush_cjk(&mut cjk, &mut tokens, &mut offset);
        total = total.saturating_add(offset);
    }
    (tokens, total)
}

pub(crate) fn group_positions(tokens: Vec<TokenOccurrence>) -> BTreeMap<String, Vec<u32>> {
    let mut grouped = BTreeMap::<String, Vec<u32>>::new();
    for token in tokens {
        grouped.entry(token.term).or_default().push(token.position);
    }
    grouped
}

pub fn normalize_term(term: &str) -> String {
    term.to_lowercase()
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3400..=0x4DBF |
        0x4E00..=0x9FFF |
        0xF900..=0xFAFF |
        0x3040..=0x30FF |
        0xAC00..=0xD7AF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_words_and_cjk_bigrams() {
        let fields = vec!["Hello, WORLD!".to_owned(), "全文检索".to_owned()];
        let (tokens, len) = tokenize_fields(&fields);
        let terms: Vec<_> = tokens.into_iter().map(|token| token.term).collect();
        assert_eq!(terms, ["hello", "world", "全文", "文检", "检索"]);
        assert_eq!(len, 5);
    }
}
