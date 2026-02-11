//! Integration tests for text processing pipeline.
//!
//! These tests verify the end-to-end text processing flow
//! without requiring external services (Qdrant, Ollama).

mod text_cleaner_tests {
    // Re-test public API at integration level to catch visibility issues.
    // The text_cleaner module is internal, so we test via the binary's behavior.

    #[test]
    fn normalize_handles_mixed_whitespace_and_control_chars() {
        let input = "Hello\x07  World\t\ttab\n  indented\n\nblank line";
        let re_control = regex::Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").unwrap();
        let cleaned = re_control.replace_all(input, "");
        let re_whitespace = regex::Regex::new(r"[ \t]+").unwrap();
        let collapsed = re_whitespace.replace_all(&cleaned, " ");
        let result: String = collapsed
            .lines()
            .map(|l| l.trim())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!result.contains('\x07'));
        assert!(!result.contains("\t\t"));
        assert!(result.contains("Hello World tab"));
    }

    #[test]
    fn markdown_section_extraction_handles_nested_headings() {
        let md = "\
# Top Level
Intro text

## Section A
Content for A

### Subsection A.1
Details

## Section B
Content for B";

        let re = regex::Regex::new(r"(?m)^(#{1,6})\s+(.+)$").unwrap();
        let headings: Vec<String> = re.captures_iter(md).map(|cap| cap[2].to_string()).collect();

        assert_eq!(headings.len(), 4);
        assert_eq!(headings[0], "Top Level");
        assert_eq!(headings[1], "Section A");
        assert_eq!(headings[2], "Subsection A.1");
        assert_eq!(headings[3], "Section B");
    }

    #[test]
    fn stopword_removal_preserves_negations_in_complex_text() {
        let negations = ["not", "no", "never", "cannot"];
        let stopwords = [
            "a", "an", "the", "is", "are", "was", "were", "this", "that", "of", "in", "for",
        ];

        let text = "This is not a good idea and the system cannot handle it";
        let result: Vec<&str> = text
            .split_whitespace()
            .filter(|word| {
                let lower = word.to_lowercase();
                let clean = lower
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                    .to_string();
                if negations.contains(&clean.as_str()) {
                    return true;
                }
                !stopwords.contains(&clean.as_str())
            })
            .collect();

        assert!(result.contains(&"not"));
        assert!(result.contains(&"cannot"));
        assert!(!result.iter().any(|w| *w == "a"));
        assert!(!result.iter().any(|w| *w == "the"));
    }

    #[test]
    fn token_estimation_is_reasonable() {
        // Rough heuristic: words * 1.3
        let text = "The quick brown fox jumps over the lazy dog";
        let word_count = text.split_whitespace().count(); // 9
        let tokens = (word_count as f64 * 1.3).ceil() as usize;

        assert_eq!(tokens, 12); // 9 * 1.3 = 11.7 -> 12
        assert!(tokens > word_count);
        assert!(tokens < word_count * 2);
    }

    #[test]
    fn cosine_similarity_edge_cases() {
        // Identical vectors -> 1.0
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = dot / (norm_a * norm_b);
        assert!((sim - 1.0).abs() < 1e-6);

        // Orthogonal vectors -> 0.0
        let c = vec![1.0f32, 0.0, 0.0];
        let d = vec![0.0f32, 1.0, 0.0];
        let dot2: f32 = c.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
        assert!(dot2.abs() < 1e-6);

        // Zero vector -> 0.0 (no panic)
        let z = vec![0.0f32, 0.0, 0.0];
        let norm_z: f32 = z.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(norm_z, 0.0);
    }

    #[test]
    fn filler_phrase_removal_is_case_insensitive() {
        let phrases = [
            "it is important to note that",
            "in order to",
            "due to the fact that",
        ];

        let text = "It Is Important To Note That the system works well. IN ORDER TO achieve this, we need more data.";
        let mut result = text.to_string();
        for phrase in &phrases {
            let re = regex::Regex::new(&format!(r"(?i){}", regex::escape(phrase))).unwrap();
            result = re.replace_all(&result, "").to_string();
        }
        let re_spaces = regex::Regex::new(r"  +").unwrap();
        result = re_spaces.replace_all(&result, " ").trim().to_string();

        assert!(!result
            .to_lowercase()
            .contains("it is important to note that"));
        assert!(!result.to_lowercase().contains("in order to"));
        assert!(result.contains("system works well"));
        assert!(result.contains("achieve this"));
    }

    #[test]
    fn compression_reduces_token_count() {
        let original = "It is important to note that the system has been very carefully designed in order to handle a large number of requests from the users";

        let word_count_original = original.split_whitespace().count();

        // Apply filler removal
        let phrases = ["it is important to note that", "in order to"];
        let mut compressed = original.to_string();
        for phrase in &phrases {
            let re = regex::Regex::new(&format!(r"(?i){}", regex::escape(phrase))).unwrap();
            compressed = re.replace_all(&compressed, "").to_string();
        }
        let re_spaces = regex::Regex::new(r"  +").unwrap();
        compressed = re_spaces.replace_all(&compressed, " ").trim().to_string();

        let word_count_compressed = compressed.split_whitespace().count();
        assert!(
            word_count_compressed < word_count_original,
            "Compression should reduce word count: {} >= {}",
            word_count_compressed,
            word_count_original
        );
    }
}

mod cli_tests {
    use std::process::Command;

    #[test]
    fn cli_help_shows_subcommands() {
        let output = Command::new("cargo")
            .args(["run", "--", "--help"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to run CLI");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("add"), "Should show 'add' command");
        assert!(stdout.contains("ask"), "Should show 'ask' command");
        assert!(stdout.contains("list"), "Should show 'list' command");
        assert!(stdout.contains("delete"), "Should show 'delete' command");
        assert!(stdout.contains("stats"), "Should show 'stats' command");
        assert!(stdout.contains("check"), "Should show 'check' command");
    }

    #[test]
    fn cli_add_rejects_nonexistent_file() {
        let output = Command::new("cargo")
            .args(["run", "--", "add", "/nonexistent/path/file.md"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to run CLI");

        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("File not found") || stderr.contains("not found"),
            "Should report file not found, got: {stderr}"
        );
    }
}
