//! The one place that holds every knob. The CLI fills this from arguments; the
//! GUI fills it from its Setup tab. The engine only ever sees a `Config`.

use std::path::PathBuf;

/// All the knobs for one listening session.
#[derive(Clone)]
pub struct Config {
    pub device: Option<String>, // substring to match, or None = system default
    pub model: PathBuf,         // whisper ggml model file
    pub whisper_bin: String,    // binary name/path for whisper-cli
    pub max_seconds: u32,       // hard cap: force a cut after this many seconds
    pub min_seconds: f32,       // do not cut a chunk shorter than this
    pub silence_ms: u32,        // trailing pause length that triggers a cut
    pub ollama_model: String,   // model name for the summary
    pub summarize: bool,        // run the summary step on exit? (CLI only)
    pub separate: bool,         // split into "Me" / "Them" tracks by channel?
    pub me_channels: Option<Vec<usize>>, // channels that carry your voice
    pub them_channels: Option<Vec<usize>>, // channels that carry the far side
}

impl Default for Config {
    fn default() -> Self {
        Config {
            device: None,
            model: PathBuf::from("models/ggml-base.en.bin"),
            whisper_bin: "whisper-cli".to_string(),
            max_seconds: 15,
            min_seconds: 3.0,
            silence_ms: 700,
            ollama_model: "llama3.1:8b".to_string(),
            summarize: true,
            separate: false,
            me_channels: None,
            them_channels: None,
        }
    }
}
