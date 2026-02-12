/// Application state for the TUI chat interface.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
pub struct DistillStats {
    pub chunks_retrieved: usize,
    pub after_dedup: usize,
    pub compression_pct: f64,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub stats: Option<DistillStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppPhase {
    Idle,
    Distilling,
    Streaming,
}

pub struct App {
    pub messages: Vec<ChatMessage>,
    pub phase: AppPhase,
    pub input: String,
    pub cursor_pos: usize,
    pub scroll_offset: u16,
    pub model_name: String,
    pub budget: Option<usize>,
    pub chunk_count: u64,
    pub tick_count: u64,
    pub ollama_ok: bool,
    pub should_quit: bool,
}

impl App {
    pub fn new(model_name: String, budget: Option<usize>) -> Self {
        Self {
            messages: Vec::new(),
            phase: AppPhase::Idle,
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            model_name,
            budget,
            chunk_count: 0,
            tick_count: 0,
            ollama_ok: false,
            should_quit: false,
        }
    }

    pub fn push_message(&mut self, role: Role, content: String, stats: Option<DistillStats>) {
        self.messages.push(ChatMessage {
            role,
            content,
            stats,
        });
        self.scroll_offset = 0;
    }

    pub fn append_to_last(&mut self, token: &str) {
        if let Some(msg) = self.messages.last_mut() {
            msg.content.push_str(token);
        }
    }

    // --- Input buffer operations ---

    pub fn insert_char(&mut self, c: char) {
        let byte_idx = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.insert(byte_idx, c);
        self.cursor_pos += 1;
    }

    pub fn delete_char_before(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            let byte_idx = self
                .input
                .char_indices()
                .nth(self.cursor_pos)
                .map(|(i, _)| i)
                .unwrap_or(self.input.len());
            self.input.remove(byte_idx);
        }
    }

    pub fn move_cursor_left(&mut self) {
        self.cursor_pos = self.cursor_pos.saturating_sub(1);
    }

    pub fn move_cursor_right(&mut self) {
        let len = self.input.chars().count();
        if self.cursor_pos < len {
            self.cursor_pos += 1;
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.chars().count();
    }

    pub fn take_input(&mut self) -> String {
        self.cursor_pos = 0;
        std::mem::take(&mut self.input)
    }
}
