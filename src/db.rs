use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    ScalarQuantizationBuilder, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde_json::Value;
use std::collections::HashMap;

pub const COLLECTION_NAME: &str = "ghost_library";
const VECTOR_DIM: u64 = 384; // MultilingualE5Small

fn qdrant_grpc_url() -> String {
    std::env::var("GHOST_QDRANT_GRPC_URL").unwrap_or_else(|_| "http://localhost:6334".to_string())
}

fn qdrant_rest_url() -> String {
    std::env::var("GHOST_QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string())
}

pub async fn create_client() -> Result<Qdrant> {
    let client = Qdrant::from_url(&qdrant_grpc_url())
        .build()
        .context("Failed to connect to Qdrant")?;
    Ok(client)
}

pub async fn ensure_collection(client: &Qdrant) -> Result<()> {
    let collections = client.list_collections().await?;
    let exists = collections
        .collections
        .iter()
        .any(|c| c.name == COLLECTION_NAME);

    if !exists {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine))
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await
            .context("Failed to create collection")?;
        println!("Created collection: {COLLECTION_NAME}");
    }
    Ok(())
}

pub async fn upsert_points(client: &Qdrant, points: Vec<PointStruct>) -> Result<()> {
    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), points))
        .await
        .context("Failed to upsert points")?;
    Ok(())
}

pub async fn search_vectors(
    client: &Qdrant,
    query_vector: Vec<f32>,
    limit: u64,
) -> Result<Vec<(f64, HashMap<String, Value>)>> {
    let results = client
        .search_points(
            SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit).with_payload(true),
        )
        .await
        .context("Failed to search points")?;

    let mut out = Vec::new();
    for point in results.result {
        let score = point.score as f64;
        let payload: HashMap<String, Value> = point
            .payload
            .into_iter()
            .map(|(k, v)| (k, qdrant_value_to_json(v)))
            .collect();
        out.push((score, payload));
    }
    Ok(out)
}

pub async fn collection_info(client: &Qdrant) -> Result<(u64, u64)> {
    let info = client
        .collection_info(COLLECTION_NAME)
        .await
        .context("Failed to get collection info")?;

    let result = info.result.context("No collection info returned")?;
    let points = result.points_count.unwrap_or(0);
    let segments = result.segments_count as u64;
    Ok((points, segments))
}

/// List unique filenames stored in the collection
pub async fn list_filenames(client: &Qdrant) -> Result<Vec<(String, usize)>> {
    let mut filenames: HashMap<String, usize> = HashMap::new();
    let mut offset = None;

    loop {
        let mut request = ScrollPointsBuilder::new(COLLECTION_NAME)
            .limit(100)
            .with_payload(true);

        if let Some(off) = offset {
            request = request.offset(off);
        }

        let response = client.scroll(request).await.context("Failed to scroll points")?;
        let result = response.result;

        if result.is_empty() {
            break;
        }

        for point in &result {
            if let Some(val) = point.payload.get("filename") {
                if let Some(name) = qdrant_value_to_json(val.clone()).as_str().map(String::from) {
                    *filenames.entry(name).or_insert(0) += 1;
                }
            }
        }

        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }

    let mut result: Vec<(String, usize)> = filenames.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// Delete all points matching a filename
pub async fn delete_by_filename(client: &Qdrant, filename: &str) -> Result<u64> {
    // Count points first
    let mut count = 0u64;
    let mut offset = None;

    loop {
        let mut request = ScrollPointsBuilder::new(COLLECTION_NAME)
            .filter(Filter::must([Condition::matches(
                "filename",
                filename.to_string(),
            )]))
            .limit(100)
            .with_payload(false);

        if let Some(off) = offset {
            request = request.offset(off);
        }

        let response = client.scroll(request).await?;
        let result = response.result;
        count += result.len() as u64;

        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }

    // Delete by filter
    client
        .delete_points(
            DeletePointsBuilder::new(COLLECTION_NAME)
                .points(Filter::must([Condition::matches(
                    "filename",
                    filename.to_string(),
                )]))
                .wait(true),
        )
        .await
        .context("Failed to delete points")?;

    Ok(count)
}

pub async fn health_check() -> Result<bool> {
    let url = format!("{}/healthz", qdrant_rest_url());
    let resp = reqwest::get(&url).await;
    match resp {
        Ok(r) => Ok(r.status().is_success()),
        Err(_) => Ok(false),
    }
}

fn qdrant_value_to_json(v: qdrant_client::qdrant::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;
    match v.kind {
        Some(Kind::StringValue(s)) => Value::String(s),
        Some(Kind::IntegerValue(i)) => Value::Number(serde_json::Number::from(i)),
        Some(Kind::DoubleValue(d)) => serde_json::Number::from_f64(d)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(Kind::BoolValue(b)) => Value::Bool(b),
        Some(Kind::NullValue(_)) => Value::Null,
        _ => Value::Null,
    }
}
