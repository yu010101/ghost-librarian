/// TUI rendering: layout, colours, widgets.
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::app::{App, AppPhase, Role};

// ── Colour palette ──────────────────────────────────────────────
const PURPLE: Color = Color::Rgb(0x93, 0x82, 0xdc);
const CYAN: Color = Color::Rgb(0x50, 0xc8, 0xdc);
const GREEN: Color = Color::Rgb(0x50, 0xdc, 0x82);
const AMBER: Color = Color::Rgb(0xdc, 0xaa, 0x50);
const DIM: Color = Color::Rgb(0x60, 0x60, 0x70);
const BG: Color = Color::Rgb(0x1a, 0x1a, 0x2e);

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

// ── Public render entry ─────────────────────────────────────────
pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    // Background fill
    let bg_block = Block::default().style(Style::default().bg(BG));
    f.render_widget(bg_block, area);

    // 4-section vertical layout: header (3) | messages (flex) | input (3) | hints (1)
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .split(area);

    draw_header(f, app, chunks[0]);
    draw_messages(f, app, chunks[1]);
    draw_input(f, app, chunks[2]);
    draw_hints(f, app, chunks[3]);
}

// ── Header ──────────────────────────────────────────────────────
fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let chunks_label = if app.chunk_count > 0 {
        format!("{} chunks", app.chunk_count)
    } else {
        "empty".to_string()
    };

    let title = Line::from(vec![
        Span::styled(
            " Ghost Librarian",
            Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" │ ", Style::default().fg(DIM)),
        Span::styled(
            format!("model: {}", app.model_name),
            Style::default().fg(CYAN),
        ),
        Span::styled(" │ ", Style::default().fg(DIM)),
        Span::styled(format!("store: {chunks_label}"), Style::default().fg(GREEN)),
        Span::styled(" │ ", Style::default().fg(DIM)),
        Span::styled(
            if app.ollama_ok {
                "Ollama: OK".to_string()
            } else {
                "Ollama: --".to_string()
            },
            Style::default().fg(if app.ollama_ok { GREEN } else { AMBER }),
        ),
        Span::raw(" "),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PURPLE))
        .style(Style::default().bg(BG));

    let header = Paragraph::new(title).block(block);
    f.render_widget(header, area);
}

// ── Messages area ───────────────────────────────────────────────
fn draw_messages(f: &mut Frame, app: &App, area: Rect) {
    let inner_block = Block::default()
        .borders(Borders::LEFT | Borders::RIGHT)
        .border_style(Style::default().fg(PURPLE))
        .style(Style::default().bg(BG));
    let inner_area = inner_block.inner(area);
    f.render_widget(inner_block, area);

    let mut lines: Vec<Line> = Vec::new();

    // Welcome message if no messages yet
    if app.messages.is_empty() && app.phase == AppPhase::Idle {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "  Welcome to Ghost Librarian",
            Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "  Ask any question about your indexed documents.",
            Style::default().fg(DIM),
        )));
        lines.push(Line::raw(""));
    }

    for msg in &app.messages {
        // Blank line between messages
        if !lines.is_empty() {
            lines.push(Line::raw(""));
        }

        match msg.role {
            Role::User => {
                lines.push(Line::from(vec![
                    Span::styled(
                        " > ",
                        Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(&msg.content, Style::default().fg(CYAN)),
                ]));
            }
            Role::Assistant => {
                // Stats line if present
                if let Some(stats) = &msg.stats {
                    let stats_text = format!(
                        " [chunks: {}→{} dedup | {:.1}% compressed]",
                        stats.chunks_retrieved, stats.after_dedup, stats.compression_pct
                    );
                    lines.push(Line::from(Span::styled(
                        stats_text,
                        Style::default().fg(GREEN),
                    )));
                }

                lines.push(Line::from(Span::styled(
                    " Ghost Librarian:",
                    Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
                )));

                // Content lines — append cursor block if still streaming
                let content = if app.phase == AppPhase::Streaming
                    && std::ptr::eq(msg as *const _, app.messages.last().unwrap() as *const _)
                {
                    format!("{}█", msg.content)
                } else {
                    msg.content.clone()
                };

                for text_line in content.lines() {
                    lines.push(Line::from(Span::styled(
                        format!(" {text_line}"),
                        Style::default().fg(Color::White),
                    )));
                }
                // If content is empty (streaming just started), show cursor
                if content.is_empty() {
                    lines.push(Line::from(Span::styled(
                        " █",
                        Style::default().fg(Color::White),
                    )));
                }
            }
            Role::System => {
                lines.push(Line::from(Span::styled(
                    format!(" {}", msg.content),
                    Style::default().fg(AMBER),
                )));
            }
        }
    }

    // Distilling phase indicator with animated spinner
    if app.phase == AppPhase::Distilling {
        let spinner_char = SPINNER[(app.tick_count as usize / 2) % SPINNER.len()];
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {spinner_char} "),
                Style::default().fg(AMBER).add_modifier(Modifier::BOLD),
            ),
            Span::styled("Distilling context", Style::default().fg(AMBER)),
            Span::styled(spinning_dots(app.tick_count), Style::default().fg(AMBER)),
        ]));
    }

    let total_lines = lines.len() as u16;
    let visible = inner_area.height;
    let max_scroll = total_lines.saturating_sub(visible);
    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let messages = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .style(Style::default().bg(BG));

    f.render_widget(messages, inner_area);
}

// ── Input bar ───────────────────────────────────────────────────
fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PURPLE))
        .style(Style::default().bg(BG));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let prompt_span = Span::styled("> ", Style::default().fg(CYAN).add_modifier(Modifier::BOLD));
    let input_span = Span::styled(&app.input, Style::default().fg(Color::White));

    let input_line = if app.input.is_empty() && app.phase == AppPhase::Idle {
        Line::from(vec![
            prompt_span,
            Span::styled("Type your question...", Style::default().fg(DIM)),
        ])
    } else {
        Line::from(vec![prompt_span, input_span])
    };

    let input_widget = Paragraph::new(input_line).style(Style::default().bg(BG));
    f.render_widget(input_widget, inner);

    // Cursor position: "> " prefix is 2 chars wide
    if app.phase == AppPhase::Idle {
        let cursor_x = inner.x + 2 + app.cursor_pos as u16;
        let cursor_y = inner.y;
        f.set_cursor_position((cursor_x, cursor_y));
    }
}

// ── Keybinding hints bar ────────────────────────────────────────
fn draw_hints(f: &mut Frame, _app: &App, area: Rect) {
    let hints = Line::from(vec![
        Span::styled(" Enter", Style::default().fg(CYAN)),
        Span::styled(" Send ", Style::default().fg(DIM)),
        Span::styled(" Esc", Style::default().fg(CYAN)),
        Span::styled(" Quit ", Style::default().fg(DIM)),
        Span::styled(" PgUp/Dn", Style::default().fg(CYAN)),
        Span::styled(" Scroll ", Style::default().fg(DIM)),
    ]);

    let widget = Paragraph::new(hints).style(Style::default().bg(BG));
    f.render_widget(widget, area);
}

// ── Helpers ─────────────────────────────────────────────────────
fn spinning_dots(tick: u64) -> String {
    let n = ((tick / 5) % 4) as usize;
    ".".repeat(n)
}
