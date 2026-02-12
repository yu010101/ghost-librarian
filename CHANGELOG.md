# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-12

### Added

- Interactive TUI chat (`ghost-lib chat`) with ratatui + crossterm
- Real-time LLM streaming display with animated cursor
- Animated spinner during context distillation
- Keybinding hints bar in TUI
- Embedded vector store (`~/.ghost-librarian/store.json`) — no external DB required
- Parallel vector search via rayon
- Ollama connectivity check in TUI header

### Changed

- Replaced Qdrant with built-in file-based vector store (zero-config)
- Replaced direct `reqwest` usage with `ollama-rs` for health checks
- Updated README with comparison table and simplified quick start

### Removed

- `qdrant-client` dependency (and transitive tonic/prost)
- `reqwest` direct dependency
- `docker-compose.yml` (Qdrant no longer needed)
- Qdrant-related environment variables (`GHOST_QDRANT_URL`, `GHOST_QDRANT_GRPC_URL`)

## [0.1.0] - 2025-04-17

### Added

- CLI with `add`, `ask`, `list`, `delete`, `stats`, `check` commands
- Context Distillation pipeline: hybrid search, dedup, compression, budget packing
- Multilingual embeddings via `MultilingualE5Small` (384 dims)
- Streaming LLM output via Ollama
- PDF, Markdown, plain text document support
- Environment variable configuration
- Pre-flight health checks for Ollama
- Unit tests for distillation and text processing
