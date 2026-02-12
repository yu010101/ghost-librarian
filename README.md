<div align="center">

# Ghost Librarian

**Local RAG engine in Rust. No Python. No Docker. No API keys.**

Ask questions about your documents — powered by Ollama and an embedded vector store, entirely on your machine.

<!-- Record with: vhs demo.tape -->
![ghost-lib chat demo](demo.gif)

[![Crates.io](https://img.shields.io/crates/v/ghost-lib)](https://crates.io/crates/ghost-lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74%2B-orange.svg)](https://www.rust-lang.org/)

</div>

## Why?

Most RAG tools need Python, Docker, a vector database, and an API key.
Ghost Librarian needs **one binary and Ollama**.

| | Ghost Librarian | Typical Python RAG |
|---|---|---|
| Install | `cargo install ghost-lib` | pip install 15 packages + docker compose |
| Vector store | Built-in (zero-config) | External DB (Qdrant / Chroma / Pinecone) |
| Runtime deps | Ollama only | Python + Docker + vector DB + API keys |
| Cold start | Instant | 5-10 s |
| Config files | 0 | .env + docker-compose.yml + ... |

## Quick Start

```bash
cargo install ghost-lib
ollama pull llama3

# Index a document
ghost-lib add paper.pdf

# Ask from the CLI
ghost-lib ask "What is context distillation?"

# Or open the interactive TUI
ghost-lib chat
```

That's it. No Docker, no config, no `.env` file.

## Features

- **Context Distillation** — Hybrid search → dedup → compress → budget-pack for maximum answer quality
- **Interactive TUI** — ratatui-based chat with real-time LLM streaming
- **Zero-config storage** — Embedded vector store under `~/.ghost-librarian/`, no external DB
- **Multilingual** — MultilingualE5Small embeddings (EN, JA, and 90+ languages)
- **PDF / Markdown / Text** — Direct document ingestion
- **Fully offline** — Nothing leaves your machine

## How It Works

```
Document ─→ Split ─→ Embed ─→ Store (local)
                                  │
Query ─→ Embed ─→ Search ─→ Dedup ─→ Compress ─→ LLM ─→ Answer
```

**Context Distillation pipeline:**

1. Embed the query with MultilingualE5Small (384 dims, local ONNX)
2. Vector-search top-20 chunks from the embedded store
3. Hybrid scoring — 70% cosine similarity + 30% keyword TF-IDF
4. Redundancy removal — pairwise cosine dedup (threshold: 0.85)
5. Compression — filler phrase removal + stopword filtering (preserving negations)
6. Budget packing — fit chunks into a configurable token budget (default: 3000)

## Commands

```
ghost-lib add <file>       Index a document (.md, .txt, .pdf)
ghost-lib ask <query>      One-shot question (CLI output)
ghost-lib chat             Interactive TUI chat
ghost-lib list             List indexed documents
ghost-lib delete <name>    Remove a document from the index
ghost-lib stats            Show index statistics
ghost-lib check            Health check (Ollama + store)
```

## TUI Key Bindings

| Key | Action |
|-----|--------|
| Enter | Send query |
| Esc / Ctrl+C | Quit |
| PageUp / PageDown | Scroll history |
| ← → | Move cursor |
| Home / End | Jump to start / end |

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `GHOST_DATA_DIR` | `~/.ghost-librarian` | Vector store location |
| `GHOST_OLLAMA_HOST` | `http://localhost` | Ollama host |
| `GHOST_OLLAMA_PORT` | `11434` | Ollama port |
| `GHOST_MODEL` | `llama3` | Default LLM model |
| `GHOST_CHUNK_SIZE` | `2000` | Max characters per chunk |

## Building from Source

```bash
git clone https://github.com/yu010101/ghost-librarian.git
cd ghost-librarian
cargo build --release
```

## License

[MIT](LICENSE)
