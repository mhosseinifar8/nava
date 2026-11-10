//! Clause splitting and token-budget chunking for streaming TTS.

/// Splits Persian text into streaming clauses.
/// Prioritizes a short first clause (3-5 words) to achieve sub-180ms TTFB,
/// followed by natural clauses (8-14 words) split at punctuation or conjunction boundaries.
pub fn split_into_streaming_clauses(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 6 {
        return vec![text.to_string()];
    }

    let mut clauses = Vec::new();
    let mut current_words = Vec::new();

    // First clause target: 3-5 words
    let mut first_clause_done = false;

    for (idx, &word) in words.iter().enumerate() {
        current_words.push(word);

        let has_punct = word.ends_with('،')
            || word.ends_with('.')
            || word.ends_with('؛')
            || word.ends_with('؟')
            || word.ends_with('!')
            || word.ends_with(',')
            || word.ends_with('?');

        let next_is_conj = if idx + 1 < words.len() {
            let next = words[idx + 1];
            next == "و" || next == "که" || next == "اما" || next == "ولی"
        } else {
            false
        };

        if !first_clause_done {
            if current_words.len() >= 3 && (has_punct || current_words.len() >= 5) {
                clauses.push(current_words.join(" "));
                current_words.clear();
                first_clause_done = true;
            }
        } else {
            if (current_words.len() >= 8 && (has_punct || next_is_conj))
                || current_words.len() >= 14
            {
                clauses.push(current_words.join(" "));
                current_words.clear();
            }
        }
    }

    if !current_words.is_empty() {
        clauses.push(current_words.join(" "));
    }

    clauses
}

/// Chunks text into strict token budget windows (12-18 tokens) for models like Pocket-TTS.
pub fn chunk_by_token_budget(text: &str, min_tokens: usize, max_tokens: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = Vec::new();

    for word in words {
        current.push(word);
        let has_punct = word.ends_with('،')
            || word.ends_with('.')
            || word.ends_with('؛')
            || word.ends_with('؟')
            || word.ends_with('!')
            || word.ends_with(',')
            || word.ends_with('?');

        if (current.len() >= min_tokens && has_punct) || current.len() >= max_tokens {
            chunks.push(current.join(" "));
            current.clear();
        }
    }

    if !current.is_empty() {
        chunks.push(current.join(" "));
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_clauses() {
        let text = "سلام، من چشم هستم. روبه‌روی شما مسیر باز است و یک صندلی سمت راست قرار دارد.";
        let clauses = split_into_streaming_clauses(text);
        assert!(!clauses.is_empty());
        assert!(clauses[0].split_whitespace().count() <= 5);
    }

    #[test]
    fn test_token_budget() {
        let text = "یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده بیست";
        let chunks = chunk_by_token_budget(text, 12, 18);
        assert!(chunks.len() >= 2);
    }
}

