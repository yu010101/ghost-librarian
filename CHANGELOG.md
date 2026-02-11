# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-17

### Added

- CLI with `add`, `ask`, `list`, `delete`, `stats`, `check` commands
- Context Distillation pipeline: hybrid search, dedup, compression, budget packing
- Multilingual embeddings via `MultilingualE5Small` (384 dims)
- Streaming LLM output via Ollama
- Qdrant vector DB with scalar quantization
- PDF, Markdown, plain text document support
- Environment variable configuration
- Pre-flight health checks for Qdrant and Ollama
- Unit tests for distillation and text processing
