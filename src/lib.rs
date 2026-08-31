//! hey-listen library: the reusable core behind both the CLI and the GUI.
//!
//! The `engine` module captures audio, chunks it at pauses, transcribes each
//! chunk with whisper.cpp, and emits [`engine::Event`] values over a channel.
//! A front-end (CLI or GUI) drives a session and renders those events however
//! it likes. This keeps ONE source of truth for behavior; only presentation
//! differs between the two binaries.

pub mod audio;
pub mod chunker;
pub mod config;
pub mod dsp;
pub mod engine;
pub mod summarize;
pub mod whisper;
