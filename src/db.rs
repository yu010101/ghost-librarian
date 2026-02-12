use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub const COLLECTION_NAME: &str = "ghost_library";

// ── Data types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct Point {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: HashMap<String, Value>,
}

/// File-backed vector store.  All data lives in a single JSON file
/// under `~/.ghost-librarian/store.json`.  For typical document
/// collections (< 50 k chunks) this is more than fast enough and
/// removes the need for any external database.
pub struct VectorStore {
    path: PathBuf,
    pub points: Vec<Point>,
}

// ── Paths ───────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GHOST_DATA_DIR") {
        PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".ghost-librarian")
    } else {
        PathBuf::from(".ghost-librarian")
    }
}

fn store_path() -> PathBuf {
    data_dir().join("store.json")
}

// ── VectorStore impl ────────────────────────────────────────────

impl VectorStore {
    fn open() -> Result<Self> {
        let path = store_path();
        let points = if path.exists() {
            let data = fs::read_to_string(&path).context("Failed to read vector store")?;
            serde_json::from_str(&data).context("Failed to parse vector store")?
        } else {
            Vec::new()
        };
        Ok(Self { path, points })
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).context("Failed to create data directory")?;
        }
        let data = serde_json::to_string(&self.points).context("Failed to serialize store")?;
        fs::write(&self.path, data).context("Failed to write vector store")?;
        Ok(())
    }
}

// ── Public API (kept async for call-site compatibility) ─────────

pub async fn open_store() -> Result<VectorStore> {
    VectorStore::open()
}

pub async fn upsert_points(store: &mut VectorStore, points: Vec<Point>) -> Result<()> {
    store.points.extend(points);
    store.save()
}

/// Minimum cosine similarity to include in results.
const MIN_SCORE: f64 = 0.1;

pub async fn search_vectors(
    store: &VectorStore,
    query_vector: Vec<f32>,
    limit: u64,
) -> Result<Vec<(f64, HashMap<String, Value>)>> {
    // Parallel cosine similarity computation via rayon
    let mut scored: Vec<(f64, usize)> = store
        .points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let sim = cosine_similarity(&query_vector, &p.vector) as f64;
            (sim, i)
        })
        .filter(|(sim, _)| *sim > MIN_SCORE)
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit as usize);

    Ok(scored
        .into_iter()
        .map(|(score, i)| (score, store.points[i].payload.clone()))
        .collect())
}

pub async fn collection_info(store: &VectorStore) -> Result<(u64, u64)> {
    Ok((store.points.len() as u64, 1))
}

pub async fn list_filenames(store: &VectorStore) -> Result<Vec<(String, usize)>> {
    let mut filenames: HashMap<String, usize> = HashMap::new();
    for point in &store.points {
        if let Some(Value::String(name)) = point.payload.get("filename") {
            *filenames.entry(name.clone()).or_insert(0) += 1;
        }
    }
    let mut result: Vec<(String, usize)> = filenames.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

pub async fn delete_by_filename(store: &mut VectorStore, filename: &str) -> Result<u64> {
    let before = store.points.len();
    store
        .points
        .retain(|p| p.payload.get("filename").and_then(|v| v.as_str()) != Some(filename));
    let deleted = (before - store.points.len()) as u64;
    if deleted > 0 {
        store.save()?;
    }
    Ok(deleted)
}

// ── Helpers ─────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
