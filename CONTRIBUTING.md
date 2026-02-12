# Contributing

Thank you for considering contributing to Ghost Librarian!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ghost-librarian.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run checks: `cargo fmt && cargo clippy && cargo test`
6. Commit and push
7. Open a Pull Request

## Development Setup

```bash
# Install Ollama and pull a model
ollama pull llama3

# Build and test
cargo build
cargo test
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Write tests for new functionality
- Keep functions small and focused

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when applicable (`Fixes #123`)

## Reporting Issues

- Use the [GitHub issue tracker](https://github.com/yu010101/ghost-librarian/issues)
- Include steps to reproduce
- Include your OS and Rust version (`rustc --version`)
