//! Silence-based chunking (a light "voice activity detector", or VAD).
//!
//! The old approach cut audio every N seconds on a fixed clock. That splits
//! words in half at the boundary, so whisper mis-hears them. This chunker
//! instead cuts at a natural PAUSE in speech, so each chunk holds whole phrases.
//!
//! Three rules decide a cut:
//!   1. MAX length  — a hard cap. Even with no pause, flush after this many
//!      seconds so a long monologue still gets transcribed promptly.
//!   2. MIN length  — do not cut until the buffer holds at least this much
//!      audio, so we never emit tiny fragments.
//!   3. SILENCE     — once past MIN, cut as soon as we hear a long-enough pause.
//!
//! How it detects silence: it slices the incoming audio into short frames
//! (~30 ms), measures each frame's loudness (RMS), and counts how much silence
//! has piled up at the tail. Enough trailing silence + enough buffered speech
//! => cut.

use crate::dsp;

/// Owns the growing audio buffer and the pause-detection state.
pub struct Chunker {
    // Fixed settings, all measured in SAMPLES so comparisons are cheap.
    frame_len: usize,       // samples per loudness frame (~30 ms)
    min_samples: usize,     // do not cut before the buffer reaches this
    max_samples: usize,     // force a cut once the buffer reaches this
    silence_needed: usize,  // trailing silence (samples) that triggers a cut
    silence_rms: f32,       // a frame quieter than this counts as silence

    // Mutable state that changes as audio flows in.
    buffer: Vec<f32>,           // committed, whole frames waiting to be emitted
    pending: Vec<f32>,          // leftover tail shorter than one frame
    trailing_silence: usize,    // consecutive silent samples at the buffer tail
    voiced_since_emit: bool,    // did we hear real speech since the last cut?
}

impl Chunker {
    /// Build a chunker for a given sample rate.
    ///
    /// - `min_secs`    : minimum chunk length (e.g. 3.0)
    /// - `max_secs`    : hard-cap chunk length (e.g. 15)
    /// - `silence_ms`  : trailing pause that triggers a cut (e.g. 700)
    /// - `silence_rms` : loudness below this is "silence" (e.g. 0.004)
    pub fn new(
        sample_rate: u32,
        min_secs: f32,
        max_secs: u32,
        silence_ms: u32,
        silence_rms: f32,
    ) -> Self {
        let sr = sample_rate as f32;
        Chunker {
            // 30 ms frame. `max(1)` guards against absurdly low sample rates.
            frame_len: ((sr * 0.030) as usize).max(1),
            min_samples: (sr * min_secs) as usize,
            max_samples: (sample_rate * max_secs) as usize,
            silence_needed: ((sr * silence_ms as f32) / 1000.0) as usize,
            silence_rms,
            buffer: Vec::new(),
            pending: Vec::new(),
            trailing_silence: 0,
            voiced_since_emit: false,
        }
    }

    /// Feed newly captured mono samples in. Returns zero or more completed
    /// chunks (usually zero or one, but a large input could complete several).
    ///
    /// Rust note: we return `Vec<Vec<f32>>` — a list of chunks, each chunk being
    /// a list of samples. The caller transcribes each one.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut ready: Vec<Vec<f32>> = Vec::new();

        // Add the new audio to whatever partial frame was left over last time.
        self.pending.extend_from_slice(samples);

        // Process complete frames one at a time.
        while self.pending.len() >= self.frame_len {
            // Move one frame from `pending` into `buffer`.
            let frame: Vec<f32> = self.pending.drain(..self.frame_len).collect();
            let loud = dsp::rms(&frame) >= self.silence_rms;
            self.buffer.extend_from_slice(&frame);

            // Update the trailing-silence run and the "heard speech" flag.
            if loud {
                self.trailing_silence = 0;
                self.voiced_since_emit = true;
            } else {
                self.trailing_silence += frame.len();
            }

            // Decide whether to cut now.
            let hit_max = self.buffer.len() >= self.max_samples;
            let hit_pause = self.voiced_since_emit
                && self.buffer.len() >= self.min_samples
                && self.trailing_silence >= self.silence_needed;

            if hit_max || hit_pause {
                ready.push(self.take_buffer());
            }
        }

        ready
    }

    /// Emit whatever is buffered right now, even a partial frame. Call this once
    /// at shutdown so the final words are not lost. Returns `None` if empty.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        // Fold any partial-frame tail back into the buffer first.
        self.buffer.append(&mut self.pending);
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.take_buffer())
        }
    }

    /// Drain the buffer into a fresh chunk and reset the pause state for the
    /// next chunk. `std::mem::take` swaps the buffer out for an empty Vec
    /// without copying the samples.
    fn take_buffer(&mut self) -> Vec<f32> {
        self.trailing_silence = 0;
        self.voiced_since_emit = false;
        std::mem::take(&mut self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 16 kHz keeps the sample math simple: 1 second = 16000 samples.
    const SR: u32 = 16_000;

    fn loud(secs: f32) -> Vec<f32> {
        vec![0.5; (SR as f32 * secs) as usize] // rms 0.5, well above threshold
    }
    fn quiet(secs: f32) -> Vec<f32> {
        vec![0.0; (SR as f32 * secs) as usize] // rms 0.0, silence
    }

    // A pause after enough speech triggers exactly one cut.
    #[test]
    fn cuts_on_pause_after_min() {
        // min 1s, max 3s, cut on 500 ms of silence, threshold 0.01.
        let mut c = Chunker::new(SR, 1.0, 3, 500, 0.01);
        let mut out = c.push(&loud(1.5));
        out.extend(c.push(&quiet(0.6)));
        assert_eq!(out.len(), 1, "one pause should give one chunk");
        assert!(out[0].len() >= SR as usize, "chunk must be at least min length");
    }

    // With no pause, the hard cap forces a cut at max length.
    #[test]
    fn cuts_on_max_without_pause() {
        let mut c = Chunker::new(SR, 1.0, 3, 500, 0.01);
        let out = c.push(&loud(4.0)); // 4s of unbroken speech, max is 3s
        assert_eq!(out.len(), 1, "max cap should force exactly one cut");
        assert_eq!(out[0].len(), (SR * 3) as usize, "cut at the 3s cap");
    }

    // A pause before reaching min length must NOT cut.
    #[test]
    fn no_cut_below_min() {
        let mut c = Chunker::new(SR, 1.0, 3, 500, 0.01);
        // 0.3s speech + 0.6s silence = 0.9s total, under the 1s minimum. The
        // pause is long enough to cut, but the buffer is too short, so no cut.
        let mut out = c.push(&loud(0.3));
        out.extend(c.push(&quiet(0.6)));
        assert!(out.is_empty(), "too short to cut yet");
        assert!(c.flush().is_some(), "flush still returns the leftover");
    }
}
