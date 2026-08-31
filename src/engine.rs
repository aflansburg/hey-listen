//! The engine: capture -> chunk -> transcribe, emitting events as it goes.
//!
//! This is the reusable core. It does NOT print or draw. Instead it sends
//! [`Event`] values over a channel. A front-end (the CLI or the GUI) starts a
//! session, reads the events, and renders them. That is the whole point of the
//! split: one behavior, two presentations.
//!
//! THREADING (unchanged from the old CLI, just relocated):
//!
//!   [audio hardware] -> [cpal callback thread] -Vec<f32>-> channel
//!                                                             |
//!   [session thread] owns the cpal Stream (it is !Send on macOS), runs the
//!   worker loop, writes the transcript file, and emits Events.

use crate::config::Config;
use crate::{audio, chunker, dsp, summarize, whisper};
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Loudness below this counts as silence. Shared by the chunker (to find
/// pauses) and process_chunk (to skip transcribing dead air).
const SILENCE_RMS: f32 = 0.004;

/// Everything the engine tells a front-end. The CLI turns these into prints;
/// the GUI turns them into transcript lines, meters, and counters.
#[derive(Clone, Debug)]
pub enum Event {
    /// Capture started; carries the facts a UI header wants to show.
    Started {
        device: String,
        sample_rate: u32,
        channels: u16,
        tracks: Vec<String>,
        transcript_path: String,
    },
    /// An informational note: the channel mapping, or a graceful-fallback warning.
    Info(String),
    /// One transcribed line. `label` is Some("Me"/"Them") only in separate mode.
    Line { label: Option<String>, text: String },
    /// A loudness reading for a track's level meter (throttled to ~10/sec).
    Level { track: usize, rms: f32 },
    /// RMS of each RAW input channel, in order. A diagnostic for finding which
    /// channel is your mic vs the far side (throttled with the track levels).
    ChannelLevels(Vec<f32>),
    /// One chunk finished; carries how long whisper took, for a latency metric.
    Chunk { track: usize, latency_ms: u128 },
    /// A non-fatal error on one chunk. The session keeps running.
    Error(String),
    /// The session has fully stopped (final flush done). Always the last event.
    Stopped,
}

/// A running session. Dropping it stops the session and joins the thread.
pub struct Session {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    /// Where the transcript is being written, for the summary step.
    pub transcript_path: PathBuf,
}

impl Session {
    /// A clone of the stop flag, so a Ctrl-C handler can request shutdown.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Request shutdown and wait for the worker to flush and exit.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start a listening session. Returns immediately; work happens on threads and
/// is reported through `events`. Fails fast only for problems we can check here
/// (a missing model file); deeper problems arrive as `Event::Error`/`Stopped`.
pub fn start(cfg: Config, events: Sender<Event>) -> Result<Session> {
    if !cfg.model.exists() {
        anyhow::bail!(
            "model file not found: {}\nDownload one, e.g.:\n  \
             curl -L -o {} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            cfg.model.display(),
            cfg.model.display()
        );
    }

    std::fs::create_dir_all("transcripts").context("failed to create transcripts/")?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let transcript_path = PathBuf::from(format!("transcripts/transcript-{stamp}.txt"));

    let running = Arc::new(AtomicBool::new(true));
    let handle = {
        let running = running.clone();
        let transcript_path = transcript_path.clone();
        std::thread::spawn(move || session_thread(cfg, events, running, transcript_path))
    };

    Ok(Session {
        running,
        handle: Some(handle),
        transcript_path,
    })
}

/// The session thread: set up capture, run the worker, always end with Stopped.
fn session_thread(
    cfg: Config,
    events: Sender<Event>,
    running: Arc<AtomicBool>,
    transcript_path: PathBuf,
) {
    if let Err(e) = run_session(&cfg, &events, &running, &transcript_path) {
        // A fatal setup error (e.g. device not found). Report it, then stop.
        let _ = events.send(Event::Error(format!("{e:#}")));
    }
    let _ = events.send(Event::Stopped);
}

fn run_session(
    cfg: &Config,
    events: &Sender<Event>,
    running: &Arc<AtomicBool>,
    transcript_path: &Path,
) -> Result<()> {
    // Channel from the cpal callback to this thread.
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    // Build the stream HERE and keep it alive for the whole session. On macOS a
    // cpal Stream is !Send, so it must live on the thread that created it.
    let (stream, capture) =
        audio::start_capture(cfg.device.as_deref(), tx).context("failed to start capture")?;
    let _stream = stream;

    // Decide the tracks; surface warnings and the channel mapping as Info.
    let (specs, notes) = build_tracks(cfg, capture.channels);
    for note in notes {
        let _ = events.send(Event::Info(note));
    }

    let labels: Vec<String> = specs
        .iter()
        .map(|s| s.label.clone().unwrap_or_else(|| "All".to_string()))
        .collect();
    let _ = events.send(Event::Started {
        device: capture.device_name.clone(),
        sample_rate: capture.sample_rate,
        channels: capture.channels,
        tracks: labels,
        transcript_path: transcript_path.display().to_string(),
    });

    worker_loop(rx, capture, running, cfg, specs, transcript_path, events)
}

/// One transcription track: a label plus the channels that feed it. In plain
/// mode there is one track (label None, all channels). In separate mode there
/// are two: "Me" and "Them".
pub struct TrackSpec {
    pub label: Option<String>,
    pub channels: Vec<usize>,
}

/// Decide the transcription tracks from the config and the device's real
/// channel count. Returns the tracks plus human-readable notes (warnings and
/// the chosen mapping). Any problem falls back to a single plain track, so
/// separation never breaks capture.
pub fn build_tracks(cfg: &Config, channels: u16) -> (Vec<TrackSpec>, Vec<String>) {
    let mut notes = Vec::new();
    let plain = || {
        vec![TrackSpec {
            label: None,
            channels: (0..channels as usize).collect(),
        }]
    };

    if !cfg.separate {
        return (plain(), notes);
    }

    if channels < 2 {
        notes.push(format!(
            "[separate] device has {channels} channel(s); need 2+. Using one track."
        ));
        return (plain(), notes);
    }

    // Keep only in-range indices, sorted and de-duplicated.
    let clean = |v: Vec<usize>| -> Vec<usize> {
        let mut v: Vec<usize> = v.into_iter().filter(|&c| c < channels as usize).collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    // Defaults: you are channel 0; the far side is every other channel.
    let me = clean(cfg.me_channels.clone().unwrap_or_else(|| vec![0]));
    let them = clean(
        cfg.them_channels
            .clone()
            .unwrap_or_else(|| (1..channels as usize).collect()),
    );

    if me.is_empty() || them.is_empty() {
        notes.push("[separate] a channel group is empty or out of range. Using one track.".to_string());
        return (plain(), notes);
    }

    notes.push(format!("Speaker separation on: Me=ch{me:?}  Them=ch{them:?}"));
    (
        vec![
            TrackSpec {
                label: Some("Me".to_string()),
                channels: me,
            },
            TrackSpec {
                label: Some("Them".to_string()),
                channels: them,
            },
        ],
        notes,
    )
}

/// One live track: its label, its channels, its own chunker, plus per-track
/// state (a chunk counter and a private temp WAV path so two tracks never
/// clobber each other's file).
struct Track {
    label: Option<String>,
    channels: Vec<usize>,
    chunker: chunker::Chunker,
    chunk_index: u32,
    wav_path: PathBuf,
}

/// A transcribed chunk, ready to record and emit.
struct Processed {
    label: Option<String>,
    text: String,
    latency_ms: u128,
}

/// The worker loop: collect samples into chunks and transcribe each one. One
/// `Chunker` runs per track. It writes the transcript file and emits events; it
/// never prints. Non-fatal per-chunk errors become `Event::Error` and the loop
/// keeps going.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    rx: mpsc::Receiver<Vec<f32>>,
    capture: audio::CaptureConfig,
    running: &Arc<AtomicBool>,
    cfg: &Config,
    specs: Vec<TrackSpec>,
    transcript_path: &Path,
    events: &Sender<Event>,
) -> Result<()> {
    let mut transcript = OpenOptions::new()
        .create(true)
        .append(true)
        .open(transcript_path)
        .with_context(|| format!("failed to open {}", transcript_path.display()))?;

    let mut tracks: Vec<Track> = specs
        .into_iter()
        .enumerate()
        .map(|(i, spec)| Track {
            label: spec.label,
            channels: spec.channels,
            chunker: chunker::Chunker::new(
                capture.sample_rate,
                cfg.min_seconds,
                cfg.max_seconds,
                cfg.silence_ms,
                SILENCE_RMS,
            ),
            chunk_index: 0,
            wav_path: std::env::temp_dir().join(format!("hey-listen-track{i}.wav")),
        })
        .collect();

    // Throttle level-meter events so we do not flood the channel.
    let mut last_level = Instant::now();

    loop {
        // Stop pulling audio as soon as shutdown is requested, so Stop/Ctrl-C
        // exits promptly instead of draining a backlog.
        if !running.load(Ordering::SeqCst) {
            break;
        }

        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(batch) => {
                let now = Instant::now();
                let emit_level = now.duration_since(last_level) >= Duration::from_millis(100);

                // Per-raw-channel meters: a diagnostic for the channel mapping.
                if emit_level {
                    let _ = events.send(Event::ChannelLevels(dsp::channel_rms(
                        &batch,
                        capture.channels,
                    )));
                }

                for (i, track) in tracks.iter_mut().enumerate() {
                    let mono = dsp::demux(&batch, capture.channels, &track.channels);

                    if emit_level {
                        let _ = events.send(Event::Level {
                            track: i,
                            rms: dsp::rms(&mono),
                        });
                    }

                    for chunk in track.chunker.push(&mono) {
                        transcribe_and_report(
                            &chunk,
                            capture.sample_rate,
                            cfg,
                            i,
                            track,
                            &mut transcript,
                            events,
                        );
                    }
                }

                if emit_level {
                    last_level = now;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if !running.load(Ordering::SeqCst) {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush each track's leftover so the final words are not lost.
    for (i, track) in tracks.iter_mut().enumerate() {
        if let Some(chunk) = track.chunker.flush() {
            transcribe_and_report(
                &chunk,
                capture.sample_rate,
                cfg,
                i,
                track,
                &mut transcript,
                events,
            );
        }
    }

    Ok(())
}

/// Transcribe one chunk, then write + emit the result. Any failure becomes an
/// `Event::Error`; it never stops the session.
fn transcribe_and_report(
    chunk: &[f32],
    from_hz: u32,
    cfg: &Config,
    track_index: usize,
    track: &mut Track,
    transcript: &mut File,
    events: &Sender<Event>,
) {
    match process_chunk(
        chunk,
        from_hz,
        &cfg.model,
        &cfg.whisper_bin,
        &track.wav_path,
        &mut track.chunk_index,
        track.label.as_deref(),
    ) {
        Ok(Some(p)) => {
            let line = match &p.label {
                Some(name) => format!("{name}: {}", p.text),
                None => p.text.clone(),
            };
            if let Err(e) = writeln!(transcript, "{line}") {
                let _ = events.send(Event::Error(format!("transcript write failed: {e}")));
            }
            let _ = transcript.flush();
            let _ = events.send(Event::Line {
                label: p.label,
                text: p.text,
            });
            let _ = events.send(Event::Chunk {
                track: track_index,
                latency_ms: p.latency_ms,
            });
        }
        Ok(None) => {} // silence or empty text: nothing to record
        Err(e) => {
            let _ = events.send(Event::Error(format!(
                "chunk {} failed: {e:#}",
                track.chunk_index
            )));
        }
    }
}

/// Resample -> WAV -> whisper for one chunk. Returns `None` for silence or empty
/// output. Measures how long whisper took, for the latency metric.
fn process_chunk(
    mono: &[f32],
    from_hz: u32,
    model: &Path,
    whisper_bin: &str,
    wav_path: &Path,
    chunk_index: &mut u32,
    label: Option<&str>,
) -> Result<Option<Processed>> {
    *chunk_index += 1;

    if dsp::rms(mono) < SILENCE_RMS {
        return Ok(None);
    }

    let resampled = dsp::resample_linear(mono, from_hz, dsp::WHISPER_HZ);
    dsp::write_wav_16k_mono(wav_path, &resampled)?;

    let t0 = Instant::now();
    let text = whisper::transcribe(whisper_bin, model, wav_path)?;
    let latency_ms = t0.elapsed().as_millis();

    if text.is_empty() {
        return Ok(None);
    }

    Ok(Some(Processed {
        label: label.map(str::to_string),
        text,
        latency_ms,
    }))
}

/// Read the transcript back, summarize it with Ollama, and save the summary
/// next to the transcript. Returns the summary text.
pub fn summarize_transcript(transcript_path: &Path, ollama_model: &str) -> Result<String> {
    let transcript = std::fs::read_to_string(transcript_path)
        .with_context(|| format!("could not read {}", transcript_path.display()))?;

    if transcript.trim().is_empty() {
        anyhow::bail!("transcript is empty — nothing to summarize");
    }

    let summary = summarize::summarize(ollama_model, &transcript)?;

    // Save as summary-<stamp>.txt beside transcript-<stamp>.txt.
    if let Some(name) = transcript_path.file_name().and_then(|n| n.to_str()) {
        let summary_name = name.replace("transcript", "summary");
        let summary_path = transcript_path.with_file_name(summary_name);
        let _ = std::fs::write(summary_path, &summary);
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_sep(me: Option<Vec<usize>>, them: Option<Vec<usize>>) -> Config {
        Config {
            separate: true,
            me_channels: me,
            them_channels: them,
            ..Config::default()
        }
    }

    #[test]
    fn build_tracks_defaults_to_two_when_separate() {
        let (tracks, _) = build_tracks(&cfg_sep(None, None), 3);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].channels, vec![0]); // Me = ch0 by default
        assert_eq!(tracks[1].channels, vec![1, 2]); // Them = the rest
    }

    #[test]
    fn build_tracks_falls_back_to_one_on_mono() {
        // Only 1 channel: separation is impossible, so one plain track + a note.
        let (tracks, notes) = build_tracks(&cfg_sep(None, None), 1);
        assert_eq!(tracks.len(), 1);
        assert!(tracks[0].label.is_none());
        assert!(!notes.is_empty());
    }

    #[test]
    fn build_tracks_falls_back_when_group_out_of_range() {
        // Both requested channels are out of range on a 2-channel device.
        let (tracks, notes) = build_tracks(&cfg_sep(Some(vec![5]), Some(vec![6])), 2);
        assert_eq!(tracks.len(), 1);
        assert!(!notes.is_empty());
    }
}
