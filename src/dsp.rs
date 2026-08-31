//! DSP = digital signal processing. This module holds the small audio-math
//! helpers: mixing many channels down to one, changing the sample rate, and
//! writing a WAV file that whisper.cpp can read.
//!
//! Rust note: `//!` comments document the *module itself*. `///` comments
//! document the *item right below them* (a function, struct, etc.).

use anyhow::{Context, Result};
use std::path::Path;

/// Whisper.cpp requires audio at exactly 16000 Hz. This constant names that
/// requirement in one place so the rest of the code reads clearly.
pub const WHISPER_HZ: u32 = 16_000;

/// Extract ONE mono stream from just the picked channels of interleaved audio.
///
/// "Interleaved" means the samples arrive as C0,C1,C2,C0,C1,C2,... — one sample
/// per channel per frame. This function averages only the channels you pick.
///
/// That single idea covers two cases:
///   - plain mono: pick every channel to mix the whole device down to one stream.
///   - speaker separation: the aggregate device carries your mic on one channel
///     and the call output on others. Picking `[0]` gives your voice; picking
///     `[1, 2]` gives the far side.
///
/// Out-of-range indices are ignored, so a bad `--me-channels` cannot crash the
/// program.
///
/// Rust note: `&[f32]` is a "slice" — a borrowed view into a block of f32
/// values. We borrow the input (no copy) and return a freshly owned `Vec<f32>`.
pub fn demux(interleaved: &[f32], channels: u16, pick: &[usize]) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    let frames = interleaved.len() / channels;
    let mut out = Vec::with_capacity(frames);

    for f in 0..frames {
        let base = f * channels;
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for &c in pick {
            if c < channels {
                sum += interleaved[base + c];
                count += 1;
            }
        }
        // If no picked channel was valid, emit silence rather than dividing by 0.
        out.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }
    out
}

/// Resample mono audio from `from_hz` to `to_hz` using linear interpolation.
///
/// Linear interpolation is the simplest usable resampler: for each output
/// sample we find its position in the input timeline and blend the two nearest
/// input samples. For speech going to a transcription model this is good
/// enough. (A higher-quality resampler would first apply a low-pass filter to
/// avoid aliasing; the `rubato` crate does that if you want to upgrade later.)
pub fn resample_linear(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }

    // The ratio of timelines. If input is 48000 and output is 16000, then
    // `ratio` is 3.0: one output sample advances 3 input samples.
    let ratio = from_hz as f64 / to_hz as f64;

    // How many output samples the whole input maps to.
    let out_len = (input.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        // Exact (fractional) position in the input for this output sample.
        let src_pos = i as f64 * ratio;
        let left = src_pos.floor() as usize;           // index just before
        let right = (left + 1).min(input.len() - 1);   // index just after (clamped)
        let frac = (src_pos - left as f64) as f32;      // 0.0..1.0 blend weight

        // Blend: mostly `left` when frac is near 0, mostly `right` near 1.
        let sample = input[left] * (1.0 - frac) + input[right] * frac;
        out.push(sample);
    }
    out
}

/// Write mono f32 samples to a 16-bit PCM WAV file at 16 kHz.
///
/// whisper-cli reads a WAV file from disk, so we hand it one per chunk. We
/// convert each f32 sample (range roughly -1.0..1.0) to an i16 (range
/// -32768..32767), which is the standard 16-bit PCM format.
pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: WHISPER_HZ,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV at {}", path.display()))?;

    for &s in samples {
        // Clamp first so a loud spike cannot wrap around into a nasty click.
        let clamped = s.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(value)?;
    }

    // `finalize` writes the WAV header lengths. Without it the file is corrupt.
    writer.finalize().context("failed to finalize WAV")?;
    Ok(())
}

/// RMS loudness of EACH raw channel of an interleaved buffer, in order.
///
/// This is a diagnostic: it lets a UI show one meter per physical channel, so
/// you can see which channel carries your voice and which carry the far side.
/// That tells you the right `--me-channels` / `--them-channels` mapping.
pub fn channel_rms(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    let mut sum_sq = vec![0.0f32; channels];
    let frames = interleaved.len() / channels;
    for f in 0..frames {
        let base = f * channels;
        for (c, acc) in sum_sq.iter_mut().enumerate() {
            let s = interleaved[base + c];
            *acc += s * s;
        }
    }
    if frames == 0 {
        return sum_sq; // all zeros
    }
    sum_sq
        .into_iter()
        .map(|acc| (acc / frames as f32).sqrt())
        .collect()
}

/// Root-mean-square loudness of a buffer, 0.0 (silence) upward. We use this to
/// skip transcribing chunks that are basically silence, which saves work.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demux_all_channels_is_plain_mono() {
        // Two stereo frames: (0.0,1.0) and (0.5,0.5) -> averages 0.5 and 0.5.
        let stereo = [0.0, 1.0, 0.5, 0.5];
        assert_eq!(demux(&stereo, 2, &[0, 1]), vec![0.5, 0.5]);
    }

    #[test]
    fn demux_picks_only_wanted_channels() {
        // 3-channel frames: [ch0, ch1, ch2] per frame.
        // Frame 0 = (1.0, 0.0, 0.0), frame 1 = (0.0, 2.0, 4.0).
        let audio = [1.0, 0.0, 0.0, 0.0, 2.0, 4.0];
        // Pick channel 0 only -> your voice.
        assert_eq!(demux(&audio, 3, &[0]), vec![1.0, 0.0]);
        // Pick channels 1 and 2 -> averaged far side: 0.0 and 3.0.
        assert_eq!(demux(&audio, 3, &[1, 2]), vec![0.0, 3.0]);
        // An out-of-range index is ignored, not a crash: ch5 does not exist.
        assert_eq!(demux(&audio, 3, &[0, 5]), vec![1.0, 0.0]);
    }

    #[test]
    fn resample_halves_length_when_rate_halves() {
        let input = vec![0.0; 100];
        let out = resample_linear(&input, 32_000, 16_000);
        assert_eq!(out.len(), 50);
    }
}
