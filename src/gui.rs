//! hey-listen GUI: a small egui/eframe window over the same engine the CLI uses.
//!
//! Two tabs:
//!   - Setup: instructions + device/separation controls + Start.
//!   - Transcript: live lines, level meters, and metrics.
//!
//! The window never does audio or transcription itself. It starts an
//! `engine::Session`, then each frame drains the `Event` channel and draws what
//! it has seen. Heavy work stays on the engine's threads, so the UI stays smooth.

use eframe::egui;
use hey_listen::audio;
use hey_listen::config::Config;
use hey_listen::engine::{self, Event, Session};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// The Navi logo as raw RGBA, baked from assets/navi.svg at build time via
/// `include_bytes!`. Straight (unmultiplied) RGBA, 256x256, row-major. Baking
/// the bytes means the GUI needs no image-decoding dependency at runtime.
const ICON_RGBA: &[u8] = include_bytes!("../assets/icon-256.rgba");
const ICON_SIZE: usize = 256;

/// Build the window icon from the embedded bytes.
fn navi_icon() -> egui::IconData {
    egui::IconData {
        rgba: ICON_RGBA.to_vec(),
        width: ICON_SIZE as u32,
        height: ICON_SIZE as u32,
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([860.0, 640.0])
            .with_title("hey-listen")
            .with_icon(navi_icon()),
        ..Default::default()
    };
    eframe::run_native(
        "hey-listen",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

#[derive(PartialEq)]
enum Tab {
    Setup,
    Transcript,
}

struct App {
    tab: Tab,

    // --- Setup fields ---
    devices: Vec<(String, u16)>, // (name, channel count)
    device_scan_error: Option<String>,
    selected_device: String, // "" means system default
    separate: bool,
    me_channels: String,
    them_channels: String,
    model_path: String,
    ollama_model: String,
    start_error: Option<String>,

    // --- Live session ---
    session: Option<Session>,
    rx: Option<Receiver<Event>>,

    // --- Display state, fed by events ---
    lines: Vec<(Option<String>, String)>, // (label, text)
    track_labels: Vec<String>,
    levels: Vec<f32>, // one RMS per track, for meters
    channel_levels: Vec<f32>, // one RMS per RAW channel, for the mapping diagnostic
    device: String,
    sample_rate: u32,
    channels: u16,
    transcript_path: Option<PathBuf>,
    chunk_count: u64,
    word_count: u64,
    last_latency_ms: u128,
    started_at: Option<Instant>,
    status: String,
    errors: Vec<String>,

    // --- Summary ---
    summarizing: bool,
    summary: Option<String>,
    summary_rx: Option<Receiver<Result<String, String>>>,

    // The Navi logo texture, loaded lazily on the first frame.
    logo: Option<egui::TextureHandle>,
}

impl App {
    fn new() -> Self {
        let defaults = Config::default();
        let mut app = App {
            tab: Tab::Setup,
            devices: Vec::new(),
            device_scan_error: None,
            selected_device: String::new(),
            separate: false,
            me_channels: "0".to_string(),
            them_channels: "1,2".to_string(),
            model_path: defaults.model.display().to_string(),
            ollama_model: defaults.ollama_model.clone(),
            start_error: None,
            session: None,
            rx: None,
            lines: Vec::new(),
            track_labels: Vec::new(),
            levels: Vec::new(),
            channel_levels: Vec::new(),
            device: String::new(),
            sample_rate: 0,
            channels: 0,
            transcript_path: None,
            chunk_count: 0,
            word_count: 0,
            last_latency_ms: 0,
            started_at: None,
            status: "Idle".to_string(),
            errors: Vec::new(),
            summarizing: false,
            summary: None,
            summary_rx: None,
            logo: None,
        };
        app.refresh_devices();
        app
    }

    fn refresh_devices(&mut self) {
        match audio::input_devices() {
            Ok(list) => {
                self.devices = list;
                self.device_scan_error = None;
            }
            Err(e) => self.device_scan_error = Some(format!("{e:#}")),
        }
    }

    fn is_running(&self) -> bool {
        self.session.is_some()
    }

    fn start_listening(&mut self) {
        self.start_error = None;

        // Parse channel lists only when separation is on.
        let (me, them) = if self.separate {
            match (
                parse_channels(&self.me_channels),
                parse_channels(&self.them_channels),
            ) {
                (Ok(m), Ok(t)) => (Some(m), Some(t)),
                _ => {
                    self.start_error = Some(
                        "Channel lists must be comma-separated numbers, e.g. 0 and 1,2".into(),
                    );
                    return;
                }
            }
        } else {
            (None, None)
        };

        let cfg = Config {
            device: if self.selected_device.is_empty() {
                None
            } else {
                Some(self.selected_device.clone())
            },
            model: PathBuf::from(self.model_path.trim()),
            ollama_model: self.ollama_model.trim().to_string(),
            summarize: false, // the GUI summarizes on demand, via a button
            separate: self.separate,
            me_channels: me,
            them_channels: them,
            ..Config::default()
        };

        let (tx, rx) = mpsc::channel::<Event>();
        match engine::start(cfg, tx) {
            Ok(session) => {
                // Reset display for the new session.
                self.transcript_path = Some(session.transcript_path.clone());
                self.session = Some(session);
                self.rx = Some(rx);
                self.lines.clear();
                self.levels.clear();
                self.channel_levels.clear();
                self.track_labels.clear();
                self.chunk_count = 0;
                self.word_count = 0;
                self.last_latency_ms = 0;
                self.errors.clear();
                self.summary = None;
                self.started_at = Some(Instant::now());
                self.status = "Listening".to_string();
                self.tab = Tab::Transcript;
            }
            Err(e) => self.start_error = Some(format!("{e:#}")),
        }
    }

    fn stop_listening(&mut self) {
        // Dropping the Session sets its stop flag and joins the worker.
        self.session = None;
        self.rx = None;
        self.status = "Idle".to_string();
        for l in self.levels.iter_mut() {
            *l = 0.0;
        }
        for l in self.channel_levels.iter_mut() {
            *l = 0.0;
        }
    }

    /// Pull every pending event and fold it into display state.
    fn drain_events(&mut self) {
        let Some(rx) = &self.rx else { return };
        // Collect first to avoid borrowing self while mutating it.
        let events: Vec<Event> = rx.try_iter().collect();
        for ev in events {
            match ev {
                Event::Started {
                    device,
                    sample_rate,
                    channels,
                    tracks,
                    transcript_path,
                } => {
                    self.device = device;
                    self.sample_rate = sample_rate;
                    self.channels = channels;
                    self.levels = vec![0.0; tracks.len()];
                    self.track_labels = tracks;
                    self.transcript_path = Some(PathBuf::from(transcript_path));
                }
                Event::Info(msg) => {
                    self.errors.push(msg);
                    if self.errors.len() > 100 {
                        self.errors.drain(0..self.errors.len() - 100);
                    }
                }
                Event::Line { label, text } => {
                    self.word_count += text.split_whitespace().count() as u64;
                    self.lines.push((label, text));
                    // Cap history so a long call cannot grow memory without bound.
                    if self.lines.len() > 5000 {
                        self.lines.drain(0..self.lines.len() - 5000);
                    }
                }
                Event::Level { track, rms } => {
                    if let Some(slot) = self.levels.get_mut(track) {
                        *slot = rms;
                    }
                }
                Event::ChannelLevels(levels) => self.channel_levels = levels,
                Event::Chunk { latency_ms, .. } => {
                    self.chunk_count += 1;
                    self.last_latency_ms = latency_ms;
                }
                Event::Error(msg) => {
                    self.errors.push(msg);
                    if self.errors.len() > 100 {
                        self.errors.drain(0..self.errors.len() - 100);
                    }
                }
                Event::Stopped => {
                    // The engine finished (Ctrl-C from elsewhere, or an error).
                    self.session = None;
                    self.rx = None;
                    self.status = "Stopped".to_string();
                }
            }
        }
    }

    fn start_summary(&mut self) {
        let Some(path) = self.transcript_path.clone() else {
            return;
        };
        let model = self.ollama_model.trim().to_string();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = engine::summarize_transcript(&path, &model).map_err(|e| format!("{e:#}"));
            let _ = tx.send(res);
        });
        self.summary_rx = Some(rx);
        self.summarizing = true;
    }

    fn poll_summary(&mut self) {
        let Some(rx) = &self.summary_rx else { return };
        if let Ok(res) = rx.try_recv() {
            self.summarizing = false;
            self.summary_rx = None;
            match res {
                Ok(s) => self.summary = Some(s),
                Err(e) => {
                    self.errors.push(format!("summary failed: {e}"));
                    if self.errors.len() > 100 {
                        self.errors.drain(0..self.errors.len() - 100);
                    }
                }
            }
        }
    }
}

impl eframe::App for App {
    // eframe 0.35: the app draws into a root `Ui` (panels take a `&mut Ui`).
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.poll_summary();

        // Load the logo texture once (from the same bytes as the window icon).
        if self.logo.is_none() {
            let img = egui::ColorImage::from_rgba_unmultiplied([ICON_SIZE, ICON_SIZE], ICON_RGBA);
            self.logo = Some(ui.ctx().load_texture("navi", img, egui::TextureOptions::LINEAR));
        }

        // Tab bar, with the Navi logo and app name on the left.
        ui.horizontal(|ui| {
            if let Some(tex) = &self.logo {
                let sized = egui::load::SizedTexture::new(tex.id(), egui::vec2(22.0, 22.0));
                ui.image(sized);
            }
            ui.label(egui::RichText::new("hey-listen").strong());
            ui.separator();
            ui.selectable_value(&mut self.tab, Tab::Setup, "Setup");
            ui.selectable_value(&mut self.tab, Tab::Transcript, "Transcript");
            ui.separator();
            ui.label(if self.is_running() {
                "🔴 Listening"
            } else {
                "⚪ Idle"
            });
        });
        ui.separator();

        match self.tab {
            Tab::Setup => self.ui_setup(ui),
            Tab::Transcript => self.ui_transcript(ui),
        }

        // Keep meters and the timer live while a session runs.
        if self.is_running() || self.summarizing {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }
}

impl App {
    fn ui_setup(&mut self, ui: &mut egui::Ui) {
        ui.heading("Setup");
        ui.add_space(4.0);

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label("Before you start:");
            ui.label("• Set the call app's OUTPUT to 'Call Output' (BlackHole + AirPods).");
            ui.label("• Set the call app's MIC to your AirPods (or built-in mic).");
            ui.label("• Pick 'Call Input' as the device below.");
            ui.label("• 'Call Input' must be an Aggregate of your mic + BlackHole 2ch.");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Input device:");
            let current = if self.selected_device.is_empty() {
                "System default".to_string()
            } else {
                self.selected_device.clone()
            };
            egui::ComboBox::from_id_salt("device")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.selected_device, String::new(), "System default");
                    for (name, ch) in &self.devices {
                        ui.selectable_value(
                            &mut self.selected_device,
                            name.clone(),
                            format!("{name}  ({ch} ch)"),
                        );
                    }
                });
            if ui.button("Refresh").clicked() {
                self.refresh_devices();
            }
        });
        if let Some(e) = &self.device_scan_error {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("device scan: {e}"));
        }

        ui.add_space(8.0);
        ui.checkbox(
            &mut self.separate,
            "Separate speakers (label Me / Them by channel)",
        );
        if self.separate {
            ui.horizontal(|ui| {
                ui.label("My channels:");
                ui.add(egui::TextEdit::singleline(&mut self.me_channels).desired_width(80.0));
                ui.label("Their channels:");
                ui.add(egui::TextEdit::singleline(&mut self.them_channels).desired_width(80.0));
            });
            ui.label(
                egui::RichText::new(
                    "Channel order follows the aggregate's sub-device order. If labels come out swapped, swap these.",
                )
                .small()
                .weak(),
            );
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Model path:");
            ui.add(egui::TextEdit::singleline(&mut self.model_path).desired_width(360.0));
        });
        ui.horizontal(|ui| {
            ui.label("Ollama model:");
            ui.add(egui::TextEdit::singleline(&mut self.ollama_model).desired_width(200.0));
        });

        ui.add_space(12.0);
        if self.is_running() {
            if ui.button("⏹ Stop").clicked() {
                self.stop_listening();
            }
        } else if ui.button("▶ Start Listening").clicked() {
            self.start_listening();
        }
        if let Some(e) = &self.start_error {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        }
    }

    fn ui_transcript(&mut self, ui: &mut egui::Ui) {
        // Metrics header.
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                metric(ui, "Status", &self.status);
                let dev = if self.device.is_empty() {
                    "—"
                } else {
                    self.device.as_str()
                };
                metric(ui, "Device", dev);
                metric(ui, "Rate", &format!("{} Hz", self.sample_rate));
                metric(ui, "Channels", &self.channels.to_string());
                metric(ui, "Elapsed", &elapsed(self.started_at));
                metric(ui, "Chunks", &self.chunk_count.to_string());
                metric(ui, "Words", &self.word_count.to_string());
                metric(ui, "Last latency", &format!("{} ms", self.last_latency_ms));
            });
        });

        // Level meters, one per track (Me / Them).
        if !self.levels.is_empty() {
            ui.add_space(4.0);
            for (i, level) in self.levels.iter().enumerate() {
                let label = self
                    .track_labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("track {i}"));
                let frac = (level * 10.0).clamp(0.0, 1.0);
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(360.0)
                        .text(label),
                );
            }
        }

        // Raw per-channel meters: a diagnostic to find the mic vs the far side.
        // Speak with the far side muted: the channel that lights up is your mic.
        if !self.channel_levels.is_empty() {
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Raw channels (find your mic):").small().weak());
            for (i, level) in self.channel_levels.iter().enumerate() {
                let frac = (level * 10.0).clamp(0.0, 1.0);
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(260.0)
                        .text(format!("ch{i}")),
                );
            }
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if self.is_running() {
                if ui.button("⏹ Stop").clicked() {
                    self.stop_listening();
                }
            } else if ui.button("▶ Start").clicked() {
                self.start_listening();
            }
            let can_sum = self.transcript_path.is_some() && !self.summarizing;
            if ui
                .add_enabled(can_sum, egui::Button::new("📝 Summarize"))
                .clicked()
            {
                self.start_summary();
            }
            if ui.button("🧹 Clear").clicked() {
                self.lines.clear();
                self.word_count = 0;
            }
            if self.summarizing {
                ui.spinner();
                ui.label("summarizing…");
            }
        });

        // Latest error, if any (non-fatal).
        if let Some(err) = self.errors.last() {
            ui.colored_label(egui::Color32::from_rgb(230, 170, 90), err);
        }

        ui.separator();

        // The live transcript, colored by speaker.
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (label, text) in &self.lines {
                    match label.as_deref() {
                        Some("Me") => ui.colored_label(
                            egui::Color32::from_rgb(120, 180, 255),
                            format!("Me: {text}"),
                        ),
                        Some("Them") => ui.colored_label(
                            egui::Color32::from_rgb(150, 220, 150),
                            format!("Them: {text}"),
                        ),
                        Some(other) => ui.label(format!("{other}: {text}")),
                        None => ui.label(text),
                    };
                }
                // Show the summary at the bottom when present.
                if let Some(summary) = &self.summary {
                    ui.add_space(10.0);
                    ui.separator();
                    ui.heading("Summary");
                    ui.label(summary);
                }
            });
    }
}

/// Draw one "Label: value" metric chip.
fn metric(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.strong(format!("{label}:"));
        ui.label(value);
    });
    ui.add_space(8.0);
}

/// Format elapsed time since `start` as M:SS.
fn elapsed(start: Option<Instant>) -> String {
    match start {
        Some(t) => {
            let s = t.elapsed().as_secs();
            format!("{}:{:02}", s / 60, s % 60)
        }
        None => "—".to_string(),
    }
}

/// Parse a channel list like "1,2" into `[1, 2]`.
fn parse_channels(s: &str) -> Result<Vec<usize>, std::num::ParseIntError> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>())
        .collect()
}
