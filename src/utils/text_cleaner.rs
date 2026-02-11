use regex::Regex;

/// Negation words to preserve during stopword removal
const NEGATIONS: &[&str] = &[
    "not",
    "no",
    "nor",
    "never",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    "none",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
];

const STOPWORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "am",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "to",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "and",
    "but",
    "or",
    "so",
    "if",
    "then",
    "because",
    "as",
    "until",
    "while",
    "about",
    "against",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "also",
    "both",
    "how",
    "when",
    "where",
    "why",
    "all",
    "any",
    "here",
    "there",
    "up",
    "out",
    "over",
    "under",
    "again",
    "further",
    "once",
];

/// Filler phrases to remove during compression
const FILLER_PHRASES: &[&str] = &[
    "it is important to note that",
    "it should be noted that",
    "it is worth mentioning that",
    "as a matter of fact",
    "in order to",
    "due to the fact that",
    "for the purpose of",
    "in the event that",
    "at the end of the day",
    "as previously mentioned",
    "it goes without saying",
    "needless to say",
    "in terms of",
    "with regard to",
    "with respect to",
    "on the other hand",
    "in addition to",
    "as well as",
    "in light of",
    "as a result of",
];

/// Normalize text: collapse whitespace, strip control characters
pub fn normalize(text: &str) -> String {
    let re_control = Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").unwrap();
    let cleaned = re_control.replace_all(text, "");
    let re_whitespace = Regex::new(r"[ \t]+").unwrap();
    let collapsed = re_whitespace.replace_all(&cleaned, " ");
    collapsed
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Extract markdown sections as (heading, content) pairs
pub fn extract_markdown_sections(text: &str) -> Vec<(String, String)> {
    let re = Regex::new(r"(?m)^(#{1,6})\s+(.+)$").unwrap();
    let mut sections = Vec::new();
    let mut last_heading = String::new();
    let mut last_start = 0;
    let mut found_first = false;

    for cap in re.captures_iter(text) {
        let m = cap.get(0).unwrap();
        if found_first {
            let content = text[last_start..m.start()].trim().to_string();
            sections.push((last_heading.clone(), content));
        }
        last_heading = cap[2].to_string();
        last_start = m.end();
        found_first = true;
    }

    if found_first {
        let content = text[last_start..].trim().to_string();
        sections.push((last_heading, content));
    } else if !text.trim().is_empty() {
        sections.push(("(no heading)".to_string(), text.trim().to_string()));
    }

    sections
}

/// Remove stopwords while preserving negations
pub fn remove_stopwords(text: &str) -> String {
    text.split_whitespace()
        .filter(|word| {
            let lower = word.to_lowercase();
            let clean = lower.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            if NEGATIONS.contains(&clean) {
                return true;
            }
            !STOPWORDS.contains(&clean)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Remove filler phrases from text
pub fn remove_filler_phrases(text: &str) -> String {
    let mut result = text.to_string();
    for phrase in FILLER_PHRASES {
        let re = Regex::new(&format!(r"(?i){}", regex::escape(phrase))).unwrap();
        result = re.replace_all(&result, "").to_string();
    }
    // Clean up double spaces left after removal
    let re_spaces = Regex::new(r"  +").unwrap();
    re_spaces.replace_all(&result, " ").trim().to_string()
}

/// Compress text by removing stopwords and filler phrases
pub fn compress_text(text: &str) -> String {
    let without_fillers = remove_filler_phrases(text);
    remove_stopwords(&without_fillers)
}

/// Estimate token count using words * 1.3 heuristic
pub fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    (word_count as f64 * 1.3).ceil() as usize
}

/// Calculate compression ratio
#[allow(dead_code)]
pub fn compression_ratio(original: &str, compressed: &str) -> f64 {
    let orig_tokens = estimate_tokens(original);
    let comp_tokens = estimate_tokens(compressed);
    if orig_tokens == 0 {
        return 0.0;
    }
    1.0 - (comp_tokens as f64 / orig_tokens as f64)
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let input = "Hello  \x07 World\t\ttab";
        let result = normalize(input);
        assert_eq!(result, "Hello World tab");
    }

    #[test]
    fn test_extract_markdown_sections() {
        let md = "# Title\nSome intro\n## Section A\nContent A\n## Section B\nContent B";
        let sections = extract_markdown_sections(md);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].0, "Title");
        assert_eq!(sections[1].0, "Section A");
        assert_eq!(sections[2].0, "Section B");
    }

    #[test]
    fn test_stopword_removal_preserves_negation() {
        let text = "This is not a good idea";
        let result = remove_stopwords(text);
        assert!(result.contains("not"));
        assert!(result.contains("good"));
        assert!(result.contains("idea"));
        assert!(!result.contains("This"));
    }

    #[test]
    fn test_filler_removal() {
        let text = "It is important to note that the system works well";
        let result = remove_filler_phrases(text);
        assert!(!result.contains("It is important to note that"));
        assert!(result.contains("system works well"));
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "This is a test sentence with seven words";
        let tokens = estimate_tokens(text);
        assert_eq!(tokens, 11); // 8 * 1.3 = 10.4 -> 11
    }

    #[test]
    fn test_compression_ratio() {
        let original = "This is a very important and absolutely critical document";
        let compressed = compress_text(original);
        let ratio = compression_ratio(original, &compressed);
        assert!(ratio > 0.0, "Compression ratio should be positive");
        assert!(ratio < 1.0, "Compression ratio should be less than 1.0");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }
}
