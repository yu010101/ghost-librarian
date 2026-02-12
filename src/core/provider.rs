use anyhow::{Context, Result};
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::options::GenerationOptions;
use ollama_rs::Ollama;
use std::io::Write;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

const SYSTEM_PROMPT: &str = r#"You are Ghost Librarian, a precise research assistant. Answer questions using ONLY the provided context. Follow these rules strictly:

1. Base your answer exclusively on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Quote specific passages when relevant
4. Be concise and factual — avoid speculation
5. If the context contains conflicting information, acknowledge it"#;

const DEFAULT_MODEL: &str = "llama3";

fn ollama_host() -> String {
    std::env::var("GHOST_OLLAMA_HOST").unwrap_or_else(|_| "http://localhost".to_string())
}

fn ollama_port() -> u16 {
    std::env::var("GHOST_OLLAMA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11434)
}

fn default_model() -> String {
    std::env::var("GHOST_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

fn create_ollama() -> Ollama {
    Ollama::new(ollama_host(), ollama_port())
}

/// Check if Ollama is running and accessible
pub async fn health_check() -> Result<bool> {
    let ollama = create_ollama();
    match ollama.list_local_models().await {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// List available models from Ollama
pub async fn list_models() -> Result<Vec<String>> {
    let ollama = create_ollama();
    let models = ollama
        .list_local_models()
        .await
        .context("Failed to list Ollama models")?;
    Ok(models.into_iter().map(|m| m.name).collect())
}

/// Generate a response using Ollama with streaming output
pub async fn ask_with_context(query: &str, context: &str, model: Option<&str>) -> Result<String> {
    let ollama = create_ollama();
    let model_name = model.unwrap_or(&default_model()).to_string();

    let prompt = format!(
        "CONTEXT:\n{context}\n\n---\nQUESTION: {query}\n\nProvide a precise answer based only on the context above."
    );

    let request = GenerationRequest::new(model_name, prompt)
        .system(SYSTEM_PROMPT.to_string())
        .options(
            GenerationOptions::default()
                .temperature(0.1)
                .num_predict(1024),
        );

    let mut stream = ollama
        .generate_stream(request)
        .await
        .context("Failed to connect to Ollama. Is it running? (ollama serve)")?;

    let mut full_response = String::new();

    while let Some(Ok(responses)) = stream.next().await {
        for response in responses {
            print!("{}", response.response);
            let _ = std::io::stdout().flush();
            full_response.push_str(&response.response);
        }
    }
    println!();

    Ok(full_response)
}

/// Events sent through the streaming channel
#[derive(Debug)]
pub enum StreamEvent {
    Token(String),
    Done,
    Error(String),
}

/// Return the active model name (from env or default)
pub fn active_model_name(model_override: Option<&str>) -> String {
    model_override
        .map(String::from)
        .unwrap_or_else(default_model)
}

/// Channel-based streaming: spawnable with owned parameters.
/// Sends tokens through `tx` as they arrive from Ollama.
pub async fn ask_with_context_stream(
    query: String,
    context: String,
    model: Option<String>,
    tx: mpsc::UnboundedSender<StreamEvent>,
) {
    let ollama = create_ollama();
    let model_name = model.unwrap_or_else(default_model);

    let prompt = format!(
        "CONTEXT:\n{context}\n\n---\nQUESTION: {query}\n\nProvide a precise answer based only on the context above."
    );

    let request = GenerationRequest::new(model_name, prompt)
        .system(SYSTEM_PROMPT.to_string())
        .options(
            GenerationOptions::default()
                .temperature(0.1)
                .num_predict(1024),
        );

    let stream_result = ollama.generate_stream(request).await;

    match stream_result {
        Ok(mut stream) => {
            while let Some(Ok(responses)) = stream.next().await {
                for response in responses {
                    if tx.send(StreamEvent::Token(response.response)).is_err() {
                        return;
                    }
                }
            }
            let _ = tx.send(StreamEvent::Done);
        }
        Err(e) => {
            let _ = tx.send(StreamEvent::Error(format!(
                "Failed to connect to Ollama: {e}"
            )));
        }
    }
}
