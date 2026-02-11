# Ghost Librarian

Ultra-lightweight local-LLM RAG engine with **Context Distillation**.

Index your documents locally, ask questions, and get precise answers powered by Ollama + Qdrant — entirely offline, no API keys required.

## Features

- **Context Distillation** — Hybrid search (vector 70% + keyword TF-IDF 30%), redundancy removal, and text compression to maximize context quality within a token budget
- **Multilingual** — Uses `MultilingualE5Small` embeddings (384 dims) for English, Japanese, and 90+ languages
- **Fully Local** — Runs entirely on your machine with Ollama and Qdrant
- **Multiple Formats** — Supports `.md`, `.txt`, `.rst`, and `.pdf`
- **Streaming Output** — LLM responses stream token-by-token to the terminal
- **Configurable** — Override defaults via environment variables

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- [Ollama](https://ollama.ai/) (for local LLM)

### Setup

```bash
# Clone the repository
git clone https://github.com/yu01/ghost-librarian.git
cd ghost-librarian

# Start Qdrant
docker compose up -d

# Pull an LLM model
ollama pull llama3

# Build
cargo build --release

# Verify everything is running
./target/release/ghost-lib check
```

### Usage

```bash
# Add a document
ghost-lib add ./docs/my-notes.md

# Add a PDF
ghost-lib add ./papers/research.pdf

# Ask a question
ghost-lib ask "What is context distillation?"

# Ask with a specific model
ghost-lib ask "Explain the architecture" --model mistral

# List indexed documents
ghost-lib list

# Delete a document
ghost-lib delete my-notes.md

# View index stats
ghost-lib stats

# Health check
ghost-lib check
```

## Architecture

```
Document → Split → Embed → Store (Qdrant)
                                    ↓
Query → Embed → Vector Search → Hybrid Ranking
                                    ↓
                     Dedup → Compress → Pack (Budget)
                                    ↓
                     Ollama LLM → Streaming Answer
```

### Context Distillation Pipeline

1. **Embed** query with `MultilingualE5Small` (384 dims, local ONNX)
2. **Vector search** top-20 chunks from Qdrant
3. **Hybrid scoring** — 70% vector similarity + 30% keyword TF-IDF
4. **Redundancy removal** — Cosine similarity dedup (threshold: 0.85)
5. **Text compression** — Filler phrase removal + stopword filtering (preserving negations)
6. **Budget packing** — Fit compressed chunks into configurable token budget (default: 3000)

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GHOST_QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `GHOST_QDRANT_GRPC_URL` | `http://localhost:6334` | Qdrant gRPC URL |
| `GHOST_OLLAMA_HOST` | `http://localhost` | Ollama host |
| `GHOST_OLLAMA_PORT` | `11434` | Ollama port |
| `GHOST_MODEL` | `llama3` | Default LLM model |
| `GHOST_CHUNK_SIZE` | `2000` | Max characters per chunk |

Example:

```bash
GHOST_MODEL=mistral GHOST_CHUNK_SIZE=1500 ghost-lib ask "What is RAG?"
```

## Development

```bash
# Run tests
cargo test

# Run with clippy lints
cargo clippy

# Format code
cargo fmt

# Build optimized release binary
cargo build --release
```

## License

[MIT](LICENSE)
