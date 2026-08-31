//! Transcription by shelling out to the `whisper-cli` binary from Homebrew's
//! `whisper-cpp`. We chose "shell out" over an in-process binding because it is
//! the easiest thing to read and debug: you can run the exact same command by
//! hand to see what happened.

use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::process::Command;

/// Run whisper-cli on one WAV file and return the transcribed text.
///
/// - `binary`   : usually "whisper-cli" (on PATH from `brew install whisper-cpp`)
/// - `model`    : path to a ggml model file, e.g. models/ggml-base.en.bin
/// - `wav_path` : the 16 kHz mono WAV we wrote for this chunk
///
/// Flags used:
///   -m <model>   which model weights to load
///   -f <file>    the input WAV
///   -nt          "no timestamps" — print plain text, no [00:00.000] prefixes
///   -np          "no prints" — suppress the progress/system log noise
///   -l auto      auto-detect language (works even with an English-only model)
pub fn transcribe(binary: &str, model: &Path, wav_path: &Path) -> Result<String> {
    let mut cmd = Command::new(binary);
    cmd.arg("-m")
        .arg(model)
        .arg("-f")
        .arg(wav_path)
        .arg("-nt")
        .arg("-np")
        .arg("-l")
        .arg("auto");

    // Put whisper in its OWN process group. When you press Ctrl-C, the terminal
    // sends SIGINT to the whole foreground group. Without this, that signal
    // would also kill the running whisper child, and the chunk would fail with
    // "signal: 2 (SIGINT)". Detaching it lets the in-flight chunk finish; our
    // own Ctrl-C handler still stops the app.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let output = cmd
        .output() // runs to completion, captures stdout + stderr
        .with_context(|| format!("failed to launch {binary}; is whisper-cpp installed?"))?;

    if !output.status.success() {
        // whisper prints its diagnostics to stderr; surface them on failure.
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("{binary} exited with {}: {stderr}", output.status));
    }

    // The transcription text is on stdout. Trim leading/trailing whitespace.
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // whisper marks a silent or music-only chunk with a single bracketed token
    // like "[BLANK_AUDIO]" or "[ Silence ]". That is not speech, so drop it.
    if is_non_speech_marker(&text) {
        return Ok(String::new());
    }
    Ok(text)
}

/// True when the whole output is one bracketed non-speech marker.
fn is_non_speech_marker(text: &str) -> bool {
    let t = text.trim();
    t.starts_with('[') && t.ends_with(']') && !t[1..].contains('[')
}
