use anyhow::Result;
use fastembed::TextEmbedding;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::core::ingest;
use crate::db;
use crate::utils::text_cleaner;

/// Result of the distillation process
pub struct DistillResult {
    pub context: String,
    pub original_tokens: usize,
    pub distilled_tokens: usize,
    pub compression_ratio: f64,
    pub chunks_retrieved: usize,
    pub chunks_after_dedup: usize,
}

/// Context budget in estimated tokens
const DEFAULT_CONTEXT_BUDGET: usize = 3000;

/// Similarity threshold for deduplication
const DEDUP_THRESHOLD: f32 = 0.85;

/// Top-K results from vector search
const TOP_K: u64 = 20;

/// Perform context distillation: hybrid search → dedup → compress → pack
pub async fn distill(
    query: &str,
    embedder: &Arc<Mutex<TextEmbedding>>,
    client: &qdrant_client::Qdrant,
    context_budget: Option<usize>,
) -> Result<DistillResult> {
    let budget = context_budget.unwrap_or(DEFAULT_CONTEXT_BUDGET);

    // 1. Generate query embedding
    let query_embedding = ingest::embed_texts(embedder, vec![query.to_string()]).await?;
    let query_vec = query_embedding.into_iter().next().unwrap();

    // 2. Vector similarity search
    let search_results = db::search_vectors(client, query_vec.clone(), TOP_K).await?;

    if search_results.is_empty() {
        return Ok(DistillResult {
            context: String::new(),
            original_tokens: 0,
            distilled_tokens: 0,
            compression_ratio: 0.0,
            chunks_retrieved: 0,
            chunks_after_dedup: 0,
        });
    }

    // 3. Hybrid scoring: vector similarity (70%) + keyword TF-IDF (30%)
    let query_terms = extract_terms(query);
    let mut scored_chunks: Vec<ScoredChunk> = Vec::new();

    for (vector_score, payload) in &search_results {
        let text = payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let section = payload
            .get("section")
            .and_then(|v| v.as_str())
            .unwrap_or("(unknown)")
            .to_string();
        let filename = payload
            .get("filename")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let keyword_score = compute_tfidf_score(&text, &query_terms);
        let hybrid_score = vector_score * 0.7 + keyword_score * 0.3;

        scored_chunks.push(ScoredChunk {
            text,
            section,
            filename,
            score: hybrid_score,
        });
    }

    // Sort by hybrid score (descending)
    scored_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let chunks_retrieved = scored_chunks.len();

    // 4. Redundancy removal: compute pairwise cosine similarity on embeddings
    let chunk_texts: Vec<String> = scored_chunks.iter().map(|c| c.text.clone()).collect();
    let chunk_embeddings = ingest::embed_texts(embedder, chunk_texts).await?;

    let deduped = remove_redundant(&scored_chunks, &chunk_embeddings, DEDUP_THRESHOLD);
    let chunks_after_dedup = deduped.len();

    // 5. Compress text and pack into context budget
    let mut original_tokens = 0;
    let mut packed_chunks: Vec<String> = Vec::new();
    let mut current_tokens = 0;

    for chunk in &deduped {
        let orig_tokens = text_cleaner::estimate_tokens(&chunk.text);
        original_tokens += orig_tokens;

        let compressed = text_cleaner::compress_text(&chunk.text);
        let comp_tokens = text_cleaner::estimate_tokens(&compressed);

        if current_tokens + comp_tokens > budget {
            // Try to fit a truncated version
            let remaining = budget.saturating_sub(current_tokens);
            if remaining > 50 {
                let truncated = truncate_to_tokens(&compressed, remaining);
                packed_chunks.push(format!("[{}] {}", chunk.section, truncated));
            }
            break;
        }

        packed_chunks.push(format!("[{}] {}", chunk.section, compressed));
        current_tokens += comp_tokens;
    }

    let context = packed_chunks.join("\n\n");
    let distilled_tokens = text_cleaner::estimate_tokens(&context);
    let compression_ratio = if original_tokens > 0 {
        1.0 - (distilled_tokens as f64 / original_tokens as f64)
    } else {
        0.0
    };

    Ok(DistillResult {
        context,
        original_tokens,
        distilled_tokens,
        compression_ratio,
        chunks_retrieved,
        chunks_after_dedup,
    })
}

struct ScoredChunk {
    text: String,
    section: String,
    #[allow(dead_code)]
    filename: String,
    score: f64,
}

/// Extract query terms for keyword matching
fn extract_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty() && w.len() > 2)
        .collect()
}

/// Compute a simple TF-IDF-like score for keyword matching
fn compute_tfidf_score(text: &str, query_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let text_lower = text.to_lowercase();
    let text_words: Vec<&str> = text_lower.split_whitespace().collect();
    let total_words = text_words.len() as f64;

    if total_words == 0.0 {
        return 0.0;
    }

    let mut score = 0.0;
    for term in query_terms {
        let count = text_words
            .iter()
            .filter(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == term.as_str())
            .count() as f64;
        // TF component (normalized by text length)
        let tf = count / total_words;
        // Simple IDF approximation (treat rarer terms as more important)
        let idf = (1.0 + count).ln() + 1.0;
        score += tf * idf;
    }

    // Normalize to 0-1 range
    (score / query_terms.len() as f64).min(1.0)
}

/// Remove redundant chunks based on cosine similarity threshold
fn remove_redundant<'a>(
    chunks: &'a [ScoredChunk],
    embeddings: &[Vec<f32>],
    threshold: f32,
) -> Vec<&'a ScoredChunk> {
    let mut kept: Vec<(usize, &ScoredChunk)> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let is_redundant = kept.iter().any(|(j, _)| {
            text_cleaner::cosine_similarity(&embeddings[i], &embeddings[*j]) > threshold
        });

        if !is_redundant {
            kept.push((i, chunk));
        }
    }

    kept.into_iter().map(|(_, c)| c).collect()
}

/// Truncate text to fit within a token budget
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let max_words = (max_tokens as f64 / 1.3).floor() as usize;
    words[..max_words.min(words.len())].join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_terms() {
        let terms = extract_terms("How does context distillation work?");
        assert!(terms.contains(&"how".to_string()));
        assert!(terms.contains(&"does".to_string()));
        assert!(terms.contains(&"context".to_string()));
        assert!(terms.contains(&"distillation".to_string()));
        assert!(terms.contains(&"work".to_string()));
    }

    #[test]
    fn test_tfidf_score() {
        let text = "Context distillation is a technique for compressing context";
        let terms = vec!["context".to_string(), "distillation".to_string()];
        let score = compute_tfidf_score(text, &terms);
        assert!(score > 0.0);
    }

    #[test]
    fn test_truncate_to_tokens() {
        let text = "This is a test sentence with several words in it";
        let truncated = truncate_to_tokens(text, 5);
        let word_count = truncated.split_whitespace().count();
        assert!(word_count <= 4); // 5 / 1.3 ≈ 3.8 → 3
    }

    #[test]
    fn test_redundancy_removal() {
        // Two identical embeddings should result in one being removed
        let chunks = vec![
            ScoredChunk {
                text: "Hello world".to_string(),
                section: "A".to_string(),
                filename: "test.md".to_string(),
                score: 0.9,
            },
            ScoredChunk {
                text: "Hello world again".to_string(),
                section: "A".to_string(),
                filename: "test.md".to_string(),
                score: 0.8,
            },
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let result = remove_redundant(&chunks, &embeddings, 0.85);
        assert_eq!(result.len(), 1);
    }
}
