//! Zero-allocation Persian script normalizer and cleaner for speech pipelines.

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    static ref RE_PREFIX_MI: Regex = Regex::new(r"(?:\b)(ن?می)\s+([آ-ی])").unwrap();
    static ref RE_SUFFIX_HA: Regex = Regex::new(r"([آ-ی])\s+(ها|های|هایم|هایت|هایش|هایمان|هایتان|هایشان)\b").unwrap();
    static ref RE_SUFFIX_TAR: Regex = Regex::new(r"([آ-ی])\s+(تر|ترین|تری)\b").unwrap();
    static ref RE_COMPOUND_ROOBEROO: Regex = Regex::new(r"\bروبهروی\b").unwrap();
    static ref RE_COMPOUND_GOFTEGOO: Regex = Regex::new(r"\bگفتوگو\b").unwrap();
    static ref RE_MULTISPACE: Regex = Regex::new(r"\s+").unwrap();
    static ref RE_PUNCT: Regex = Regex::new(r"([،.؟؛!?:,;])").unwrap();
}

/// Normalizes Persian characters (Arabic Yeh/Kaf conversion, diacritic stripping, ZWNJ repairs).
pub fn normalize_persian(input: &str) -> String {
    let mut out = String::with_capacity(input.len());

    for ch in input.chars() {
        match ch {
            // Arabic Kaf -> Persian Keheh
            '\u{0643}' => out.push('\u{06A9}'),
            // Arabic Yeh / Alif Maksura -> Persian Yeh
            '\u{064A}' | '\u{0649}' => out.push('\u{06CC}'),
            // Arabic Eastern digits to ASCII / Persian
            '\u{0660}'..='\u{0669}' => {
                let digit = (ch as u32) - 0x0660;
                out.push(char::from_u32(0x06F0 + digit).unwrap_or(ch));
            }
            // Strip Tatweel (Kashida)
            '\u{0640}' => {}
            // Normalize non-breaking space to regular space
            '\u{00A0}' => out.push(' '),
            _ => out.push(ch),
        }
    }

    // Repair verbal prefixes (می‌ / نمی‌)
    let s = RE_PREFIX_MI.replace_all(&out, "$1\u{200C}$2");
    // Repair suffixes (ها / تر)
    let s = RE_SUFFIX_HA.replace_all(&s, "$1\u{200C}$2");
    let s = RE_SUFFIX_TAR.replace_all(&s, "$1\u{200C}$2");
    // Repair common compound words
    let s = RE_COMPOUND_ROOBEROO.replace_all(&s, "روبه‌روی");
    let s = RE_COMPOUND_GOFTEGOO.replace_all(&s, "گفت‌وگو");

    RE_MULTISPACE.replace_all(&s, " ").trim().to_string()
}

/// Prepares text specifically for Matcha-TTS / Sherpa-ONNX phonemizer:
/// Separates punctuation, strips unsupported kasra/ezafe, and replaces ZWNJ with space.
pub fn clean_for_matcha(input: &str) -> String {
    let norm = normalize_persian(input);
    let mut out = String::with_capacity(norm.len());

    for ch in norm.chars() {
        match ch {
            // Strip Ezafe / Kasra / Harakat that break Matcha lexicon
            '\u{0650}' | '\u{064E}' | '\u{064F}' | '\u{064B}' | '\u{064C}' | '\u{064D}' | '\u{0651}' => {}
            // Replace ZWNJ with space for Matcha tokenizer compatibility
            '\u{200C}' => out.push(' '),
            _ => out.push(ch),
        }
    }

    // Separate punctuation with spaces so they become distinct pause tokens
    let s = RE_PUNCT.replace_all(&out, " $1 ");
    RE_MULTISPACE.replace_all(&s, " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persian_normalizer() {
        let input = "سلام، كودك روبهروی شماست";
        let res = normalize_persian(input);
        assert!(res.contains("کودک"));
        assert!(res.contains("روبه‌روی"));
    }

    #[test]
    fn test_clean_for_matcha() {
        let input = "سلام، من چشم هستم. روبه‌روی شما مسیرِ باز است!";
        let cleaned = clean_for_matcha(input);
        assert!(!cleaned.contains('\u{0650}')); // No kasra
        assert!(cleaned.contains("سلام ، من چشم هستم ."));
    }
}

