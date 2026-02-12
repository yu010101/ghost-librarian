mod core;
mod db;
mod tui;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ghost-lib",
    about = "Ghost Librarian — ultra-lightweight local-LLM RAG with Context Distillation",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a document to the library (supports .md, .txt, .pdf)
    Add {
        /// Path to the document file
        path: PathBuf,
    },
    /// Ask a question using context distillation + local LLM
    Ask {
        /// Your question
        query: String,
        /// LLM model to use (default: llama3, override with GHOST_MODEL)
        #[arg(short, long)]
        model: Option<String>,
        /// Context budget in tokens (default: 3000)
        #[arg(short, long)]
        budget: Option<usize>,
    },
    /// List all indexed documents
    List,
    /// Delete an indexed document by filename
    Delete {
        /// Filename to delete (as shown in `ghost-lib list`)
        filename: String,
    },
    /// Show index statistics
    Stats,
    /// Health check for Ollama
    Check,
    /// Interactive TUI chat with context distillation
    Chat {
        /// LLM model to use (default: llama3, override with GHOST_MODEL)
        #[arg(short, long)]
        model: Option<String>,
        /// Context budget in tokens (default: 3000)
        #[arg(short, long)]
        budget: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Add { path } => cmd_add(&path).await,
        Commands::Ask {
            query,
            model,
            budget,
        } => cmd_ask(&query, model.as_deref(), budget).await,
        Commands::List => cmd_list().await,
        Commands::Delete { filename } => cmd_delete(&filename).await,
        Commands::Stats => cmd_stats().await,
        Commands::Check => cmd_check().await,
        Commands::Chat { model, budget } => tui::cmd_chat(model.as_deref(), budget).await,
    }
}

/// Pre-flight check: ensure Ollama is reachable
async fn require_ollama() -> Result<()> {
    if !core::provider::health_check().await? {
        anyhow::bail!(
            "Ollama is not reachable.\n\
             Start it with: ollama serve"
        );
    }
    Ok(())
}

async fn cmd_add(path: &std::path::Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }

    let mut store = db::open_store().await?;

    let embedder = core::ingest::create_embedder()?;
    let chunks = core::ingest::ingest_file(path, &embedder, &mut store).await?;

    println!(
        "\nSuccessfully indexed {chunks} chunks from {}",
        path.display()
    );
    Ok(())
}

async fn cmd_ask(query: &str, model: Option<&str>, budget: Option<usize>) -> Result<()> {
    require_ollama().await?;

    let store = db::open_store().await?;
    let embedder = core::ingest::create_embedder()?;

    println!("Distilling context...\n");
    let result = core::distill::distill(query, &embedder, &store, budget).await?;

    if result.context.is_empty() {
        println!("No relevant documents found. Add documents first with: ghost-lib add <path>");
        return Ok(());
    }

    println!("--- Distillation Stats ---");
    println!("  Chunks retrieved:   {}", result.chunks_retrieved);
    println!("  After dedup:        {}", result.chunks_after_dedup);
    println!("  Original tokens:    {}", result.original_tokens);
    println!("  Distilled tokens:   {}", result.distilled_tokens);
    println!(
        "  Compression:        {:.1}%",
        result.compression_ratio * 100.0
    );
    println!("--------------------------\n");

    println!("Generating answer...\n");
    core::provider::ask_with_context(query, &result.context, model).await?;

    Ok(())
}

async fn cmd_list() -> Result<()> {
    let store = db::open_store().await?;

    match db::list_filenames(&store).await {
        Ok(files) if !files.is_empty() => {
            println!("Indexed documents:\n");
            for (filename, chunks) in &files {
                println!("  {filename}  ({chunks} chunks)");
            }
            println!("\n  Total: {} document(s)", files.len());
        }
        Ok(_) => {
            println!("No documents indexed. Add one with: ghost-lib add <path>");
        }
        Err(_) => {
            println!("No documents indexed. Add one with: ghost-lib add <path>");
        }
    }

    Ok(())
}

async fn cmd_delete(filename: &str) -> Result<()> {
    let mut store = db::open_store().await?;
    let deleted = db::delete_by_filename(&mut store, filename).await?;

    if deleted > 0 {
        println!("Deleted {deleted} chunks for: {filename}");
    } else {
        println!("No chunks found for: {filename}");
        println!("Use `ghost-lib list` to see indexed documents.");
    }

    Ok(())
}

async fn cmd_stats() -> Result<()> {
    let store = db::open_store().await?;

    let (points, _segments) = db::collection_info(&store).await?;
    if points > 0 {
        println!("Ghost Library Stats");
        println!("  Collection:  {}", db::COLLECTION_NAME);
        println!("  Documents:   {points} chunks indexed");
    } else {
        println!("No documents indexed. Add one with: ghost-lib add <path>");
    }

    Ok(())
}

async fn cmd_check() -> Result<()> {
    print!("Ollama ...  ");
    match core::provider::health_check().await? {
        true => {
            println!("OK");
            match core::provider::list_models().await {
                Ok(models) if !models.is_empty() => {
                    println!("  Models: {}", models.join(", "));
                }
                Ok(_) => {
                    println!("  No models found — run: ollama pull llama3");
                }
                Err(e) => {
                    println!("  Could not list models: {e}");
                }
            }
        }
        false => println!("UNREACHABLE — run: ollama serve"),
    }

    let store = db::open_store().await?;
    let (points, _) = db::collection_info(&store).await?;
    println!("Store  ...  OK ({points} chunks)");

    Ok(())
}
