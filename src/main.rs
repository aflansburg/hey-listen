//! hey-listen CLI: parse arguments, start an engine session, print the events.
//!
//! All the real work lives in the `hey_listen` library (see `src/engine.rs`).
//! This binary is just a thin front-end: it turns command-line flags into a
//! `Config`, drives a session, and renders the event stream as text. The GUI
//! (`src/gui.rs`) drives the exact same engine.

use anyhow::{Context, Result};
use hey_listen::config::Config;
use hey_listen::engine::{self, Event};
use hey_listen::audio;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

fn print_usage() {
    println!(
        "hey-listen — local call transcription + summary

USAGE:
  hey-listen [OPTIONS]

OPTIONS:
  --list-devices          List input devices and exit
  --device <SUBSTR>       Record from the device whose name contains SUBSTR
                          (e.g. --device aggregate). Default: system default input.
  --model <PATH>          Whisper ggml model file (default: models/ggml-base.en.bin)
  --whisper-bin <NAME>    whisper binary (default: whisper-cli)
  --max-seconds <N>       Hard cap: force a cut after N seconds (default: 15)
  --min-seconds <N>       Never cut a chunk shorter than N seconds (default: 3)
  --silence-ms <N>        A pause of N ms triggers a cut (default: 700)
  --ollama-model <NAME>   Ollama model for the summary (default: llama3.1:8b)
  --no-summary            Skip the Ollama summary on exit
  --separate              Split into 'Me' / 'Them' tracks by channel.
                          Needs a 2+ channel device (e.g. mic + BlackHole).
  --me-channels <LIST>    Comma-separated channels for your voice (default: 0)
  --them-channels <LIST>  Comma-separated channels for the far side
                          (default: every channel except 0)
  -h, --help              Show this help

Press Ctrl-C to stop. The transcript and summary are saved under transcripts/."
    );
}

/// Hand-parse the arguments. We avoid a CLI crate so the parsing is visible.
fn parse_args() -> Result<Option<Config>> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1); // skip the program name

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(None); // signal "handled, do not run"
            }
            "--list-devices" => {
                audio::list_input_devices()?;
                return Ok(None);
            }
            "--device" => cfg.device = Some(next_value(&mut args, "--device")?),
            "--model" => cfg.model = PathBuf::from(next_value(&mut args, "--model")?),
            "--whisper-bin" => cfg.whisper_bin = next_value(&mut args, "--whisper-bin")?,
            "--max-seconds" => {
                cfg.max_seconds = next_value(&mut args, "--max-seconds")?
                    .parse()
                    .context("--max-seconds must be a positive integer")?;
            }
            "--min-seconds" => {
                cfg.min_seconds = next_value(&mut args, "--min-seconds")?
                    .parse()
                    .context("--min-seconds must be a number")?;
            }
            "--silence-ms" => {
                cfg.silence_ms = next_value(&mut args, "--silence-ms")?
                    .parse()
                    .context("--silence-ms must be a positive integer")?;
            }
            "--ollama-model" => cfg.ollama_model = next_value(&mut args, "--ollama-model")?,
            "--no-summary" => cfg.summarize = false,
            "--separate" => cfg.separate = true,
            "--me-channels" => {
                cfg.me_channels = Some(parse_channels(&next_value(&mut args, "--me-channels")?)?);
            }
            "--them-channels" => {
                cfg.them_channels =
                    Some(parse_channels(&next_value(&mut args, "--them-channels")?)?);
            }
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
    }
    Ok(Some(cfg))
}

/// Pull the value that follows a flag, or error if it is missing.
fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

/// Parse a channel list like "1,2" into `[1, 2]`.
fn parse_channels(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid channel number: {part:?}"))
        })
        .collect()
}

fn main() -> Result<()> {
    let cfg = match parse_args()? {
        Some(cfg) => cfg,
        None => return Ok(()),
    };

    // Remember the bits we need AFTER the session (cfg is moved into `start`).
    let want_summary = cfg.summarize;
    let ollama_model = cfg.ollama_model.clone();

    // Start the session; the engine reports everything through `rx`.
    let (tx, rx) = mpsc::channel::<Event>();
    let session = engine::start(cfg, tx).context("failed to start session")?;

    // Ctrl-C flips the session's stop flag; the worker flushes and exits.
    {
        let flag = session.stop_flag();
        ctrlc::set_handler(move || {
            eprintln!("\n(stopping — finishing the last chunk...)");
            flag.store(false, Ordering::SeqCst);
        })
        .context("failed to install Ctrl-C handler")?;
    }

    // Render the event stream until the session stops.
    for ev in &rx {
        match ev {
            Event::Started {
                device,
                sample_rate,
                channels,
                transcript_path,
                ..
            } => {
                println!("Recording from: {device}  ({sample_rate} Hz, {channels} channel(s))");
                println!("Transcript: {transcript_path}");
                println!("Listening. Press Ctrl-C to stop.\n");
            }
            Event::Info(msg) => println!("{msg}"),
            Event::Line { label, text } => match label {
                Some(name) => println!("{name}: {text}"),
                None => println!("{text}"),
            },
            Event::Error(msg) => eprintln!("[error] {msg}"),
            Event::Stopped => break,
            // metrics: ignored by the CLI
            Event::Level { .. } | Event::Chunk { .. } | Event::ChannelLevels(_) => {}
        }
    }

    // Summarize the whole transcript, if enabled.
    if want_summary {
        println!("\nSummarizing with Ollama ({ollama_model})...");
        match engine::summarize_transcript(&session.transcript_path, &ollama_model) {
            Ok(summary) => println!("\n===== SUMMARY =====\n{summary}"),
            Err(e) => eprintln!("summary failed: {e:#}"),
        }
    }

    println!("\nDone.");
    Ok(())
}
