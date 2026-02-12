/// Event loop: crossterm keyboard + LLM token channel, multiplexed with tokio::select!
use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyModifiers};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use crate::core::{distill, ingest, provider};
use crate::db;

use super::app::{App, AppPhase, DistillStats, Role};
use super::ui;

type Embedder = Arc<Mutex<fastembed::TextEmbedding>>;

/// Run the main event loop with integrated redraw. Returns when the user quits.
pub async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    let mut event_stream = EventStream::new();

    let (llm_tx, mut llm_rx) = mpsc::unbounded_channel::<provider::StreamEvent>();
    let (distill_tx, mut distill_rx) =
        mpsc::unbounded_channel::<Result<(distill::DistillResult, String), String>>();

    // Pre-flight: load store to get chunk count
    if let Ok(store) = db::open_store().await {
        let (count, _) = db::collection_info(&store).await.unwrap_or((0, 0));
        app.chunk_count = count;
    }

    // Pre-flight: check Ollama connectivity
    app.ollama_ok = provider::health_check().await.unwrap_or(false);
    if !app.ollama_ok {
        app.push_message(
            Role::System,
            "Ollama is not reachable. Start it with: ollama serve".into(),
            None,
        );
    }

    // Create embedder once (heavyweight; holds the ONNX model)
    let embedder: Option<Arc<Embedder>> = match ingest::create_embedder() {
        Ok(e) => Some(Arc::new(e)),
        Err(err) => {
            app.push_message(
                Role::System,
                format!("Warning: embedder init failed: {err}"),
                None,
            );
            None
        }
    };

    // Redraw timer — ~30 fps for smooth streaming display
    let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(33));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            // Tick → redraw + advance animation counter
            _ = tick.tick() => {
                app.tick_count = app.tick_count.wrapping_add(1);
                terminal.draw(|f| ui::draw(f, app))?;
            }

            // Keyboard events
            maybe_event = event_stream.next() => {
                let Some(Ok(event)) = maybe_event else { break };
                if let Event::Key(key) = event {
                    handle_key(app, key, &llm_tx, &distill_tx, &embedder);
                }
                if app.should_quit {
                    break;
                }
                terminal.draw(|f| ui::draw(f, app))?;
            }

            // LLM streaming tokens
            Some(stream_event) = llm_rx.recv() => {
                match stream_event {
                    provider::StreamEvent::Token(tok) => {
                        app.append_to_last(&tok);
                    }
                    provider::StreamEvent::Done => {
                        app.phase = AppPhase::Idle;
                    }
                    provider::StreamEvent::Error(e) => {
                        app.push_message(Role::System, format!("LLM error: {e}"), None);
                        app.phase = AppPhase::Idle;
                    }
                }
            }

            // Distillation results
            Some(result) = distill_rx.recv() => {
                match result {
                    Ok((dr, query)) => {
                        if dr.context.is_empty() {
                            app.push_message(
                                Role::System,
                                "No relevant documents found. Add documents first with: ghost-lib add <path>".into(),
                                None,
                            );
                            app.phase = AppPhase::Idle;
                            continue;
                        }

                        let stats = DistillStats {
                            chunks_retrieved: dr.chunks_retrieved,
                            after_dedup: dr.chunks_after_dedup,
                            compression_pct: dr.compression_ratio * 100.0,
                        };

                        app.push_message(Role::Assistant, String::new(), Some(stats));
                        app.phase = AppPhase::Streaming;

                        let tx = llm_tx.clone();
                        let context = dr.context;
                        let model = Some(app.model_name.clone());
                        tokio::spawn(async move {
                            provider::ask_with_context_stream(query, context, model, tx).await;
                        });
                    }
                    Err(e) => {
                        app.push_message(Role::System, format!("Distillation error: {e}"), None);
                        app.phase = AppPhase::Idle;
                    }
                }
            }
        }
    }

    Ok(())
}

fn handle_key(
    app: &mut App,
    key: crossterm::event::KeyEvent,
    _llm_tx: &mpsc::UnboundedSender<provider::StreamEvent>,
    distill_tx: &mpsc::UnboundedSender<Result<(distill::DistillResult, String), String>>,
    embedder: &Option<Arc<Embedder>>,
) {
    // Ctrl+C or Esc → quit
    if key.code == KeyCode::Esc
        || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c'))
    {
        app.should_quit = true;
        return;
    }

    match app.phase {
        AppPhase::Idle => match key.code {
            KeyCode::Enter => {
                let query = app.take_input().trim().to_string();
                if query.is_empty() {
                    return;
                }

                app.push_message(Role::User, query.clone(), None);
                app.phase = AppPhase::Distilling;

                let Some(embedder) = embedder.clone() else {
                    app.push_message(
                        Role::System,
                        "Embedder not available — cannot distill.".into(),
                        None,
                    );
                    app.phase = AppPhase::Idle;
                    return;
                };

                let budget = app.budget;
                let tx = distill_tx.clone();
                tokio::spawn(async move {
                    let store = match db::open_store().await {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(e.to_string()));
                            return;
                        }
                    };
                    match distill::distill(&query, &embedder, &store, budget).await {
                        Ok(result) => {
                            let _ = tx.send(Ok((result, query)));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e.to_string()));
                        }
                    }
                });
            }
            KeyCode::Char(c) => app.insert_char(c),
            KeyCode::Backspace => app.delete_char_before(),
            KeyCode::Left => app.move_cursor_left(),
            KeyCode::Right => app.move_cursor_right(),
            KeyCode::Home => app.move_cursor_home(),
            KeyCode::End => app.move_cursor_end(),
            KeyCode::PageUp => {
                app.scroll_offset = app.scroll_offset.saturating_add(5);
            }
            KeyCode::PageDown => {
                app.scroll_offset = app.scroll_offset.saturating_sub(5);
            }
            _ => {}
        },
        AppPhase::Distilling | AppPhase::Streaming => match key.code {
            KeyCode::PageUp => {
                app.scroll_offset = app.scroll_offset.saturating_add(5);
            }
            KeyCode::PageDown => {
                app.scroll_offset = app.scroll_offset.saturating_sub(5);
            }
            _ => {}
        },
    }
}
