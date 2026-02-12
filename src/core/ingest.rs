use anyhow::{bail, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use text_splitter::MarkdownSplitter;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::db;
use crate::utils::text_cleaner;

/// Create a shared embedding model (MultilingualE5Small, 384 dims — supports EN/JA/etc.)
pub fn create_embedder() -> Result<Arc<Mutex<TextEmbedding>>> {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small).with_show_download_progress(true),
    )
    .context("Failed to initialize embedding model")?;
    Ok(Arc::new(Mutex::new(model)))
}

/// Generate embeddings for texts using spawn_blocking (fastembed is not Send-safe)
pub async fn embed_texts(
    embedder: &Arc<Mutex<TextEmbedding>>,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    let embedder = embedder.clone();
    tokio::task::spawn_blocking(move || {
        let model = embedder.blocking_lock();
        model
            .embed(texts, None)
            .context("Embedding generation failed")
    })
    .await?
}

/// Read a document file and return its text content
fn read_document(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "md" | "txt" | "text" | "rst" => {
            std::fs::read_to_string(path).context("Failed to read text file")
        }
        "pdf" => {
            let bytes = std::fs::read(path).context("Failed to read PDF file")?;
            pdf_extract::extract_text_from_mem(&bytes)
                .context("Failed to extract text from PDF (scanned PDFs are not supported)")
        }
        _ => bail!("Unsupported file format: .{ext} (supported: .md, .txt, .pdf)"),
    }
}

/// Ingest a document: read, split, embed, and store
pub async fn ingest_file(
    path: &Path,
    embedder: &Arc<Mutex<TextEmbedding>>,
    store: &mut db::VectorStore,
) -> Result<usize> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("Reading: {filename}");
    let raw_text = read_document(path)?;
    let text = text_cleaner::normalize(&raw_text);

    if text.is_empty() {
        bail!("Document is empty after normalization");
    }

    // Semantic split (configurable via GHOST_CHUNK_SIZE, default 2000 chars)
    let chunk_size: usize = std::env::var("GHOST_CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let splitter = MarkdownSplitter::new(chunk_size);
    let chunks: Vec<&str> = splitter.chunks(&text).collect();
    let total_chunks = chunks.len();

    if total_chunks == 0 {
        bail!("No chunks produced from document");
    }

    println!("Split into {total_chunks} chunks");

    // Progress bar
    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    // Extract sections for metadata
    let sections = text_cleaner::extract_markdown_sections(&text);

    // Process in batches of 32
    let batch_size = 32;
    let mut all_points = Vec::new();

    for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
        let texts: Vec<String> = batch.iter().map(|s| s.to_string()).collect();
        let embeddings = embed_texts(embedder, texts.clone()).await?;

        for (i, (chunk_text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
            let chunk_index = batch_idx * batch_size + i;

            // Find the section this chunk belongs to
            let section_name = find_section_for_chunk(chunk_text, &sections);

            let payload: HashMap<String, Value> = [
                ("filename".to_string(), Value::String(filename.clone())),
                ("section".to_string(), Value::String(section_name)),
                ("chunk_index".to_string(), serde_json::json!(chunk_index)),
                ("text".to_string(), Value::String(chunk_text.clone())),
            ]
            .into_iter()
            .collect();

            let point = db::Point {
                id: Uuid::new_v4().to_string(),
                vector: embedding.clone(),
                payload,
            };
            all_points.push(point);
            pb.inc(1);
        }
    }

    // Upsert all points
    db::upsert_points(store, all_points).await?;

    pb.finish_with_message("Done");
    println!(
        "Ingested {total_chunks} chunks from {filename} ({} tokens est.)",
        text_cleaner::estimate_tokens(&text)
    );

    Ok(total_chunks)
}

/// Find which markdown section a chunk belongs to
fn find_section_for_chunk(chunk: &str, sections: &[(String, String)]) -> String {
    for (heading, content) in sections {
        if content.contains(chunk)
            || chunk.contains(content.get(..50.min(content.len())).unwrap_or(content))
        {
            return heading.clone();
        }
    }
    "(unknown)".to_string()
}
