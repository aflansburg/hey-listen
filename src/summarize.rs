//! Summarization by calling the local Ollama HTTP API.
//!
//! Ollama runs a server on http://localhost:11434. We POST to `/api/generate`
//! with a model name and a prompt, and read back the generated text. Because it
//! is local, this is free and private — the transcript never leaves your Mac.

use anyhow::{Context, Result};
use serde::Deserialize;

/// The subset of Ollama's JSON response we care about.
///
/// Rust note: `#[derive(Deserialize)]` lets `serde` build this struct straight
/// from JSON. Fields we do not list are simply ignored during parsing.
#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

/// Send the transcript to Ollama and return a summary string.
///
/// - `model`      : an installed Ollama model, e.g. "llama3.1:8b"
/// - `transcript` : the full text we collected during the call
pub fn summarize(model: &str, transcript: &str) -> Result<String> {
    // The instruction we wrap around the transcript. Keep it direct.
    let prompt = format!(
        "You are summarizing a meeting or call transcript. \
         Write a short summary, then a bullet list of key points, \
         then a bullet list of any action items or decisions. \
         If something is unclear, say so rather than inventing it.\n\n\
         Transcript:\n{transcript}"
    );

    // The request body Ollama expects. `stream: false` means "return the whole
    // answer in one JSON response" instead of a stream of partial tokens.
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
    });

    // A blocking HTTP client. We build one per call, which is fine for a single
    // request at program exit. Summaries can be slow, so allow a long timeout.
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("failed to build HTTP client")?;

    let resp = client
        .post("http://localhost:11434/api/generate")
        .json(&body)
        .send()
        .context("failed to reach Ollama at localhost:11434; is `ollama serve` running?")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("Ollama returned {status}: {text}");
    }

    let parsed: GenerateResponse = resp.json().context("failed to parse Ollama response")?;
    Ok(parsed.response.trim().to_string())
}
