//! Audio capture with `cpal`. This module finds an input device, opens a
//! stream, and pushes raw samples to the rest of the program through a channel.
//!
//! Big picture: audio hardware calls *us*. `cpal` runs a real-time callback on
//! its own high-priority thread every few milliseconds with a fresh buffer of
//! samples. We must do almost nothing in that callback — just copy the samples
//! and send them off. All the slow work (resample, WAV, whisper) happens on a
//! separate worker thread so we never stall the audio hardware.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample; // brings the `from_sample` conversion method into scope
use std::sync::mpsc::{Sender};

/// Facts about the open stream that the worker thread needs to interpret the
/// raw samples correctly. (Not `Copy`: it now owns a `String`.)
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Return each input device as (name, channel count). The channel count is the
/// one the app will ACTUALLY use (max-channel pick), which the GUI needs for its
/// device dropdown and the `--me`/`--them` channel hints.
pub fn input_devices() -> Result<Vec<(String, u16)>> {
    let host = cpal::default_host();
    let mut out = Vec::new();
    for device in host.input_devices()? {
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        let channels = pick_max_channel_config(&device)
            .map(|c| c.channels())
            .unwrap_or(0);
        out.push((name, channels));
    }
    Ok(out)
}

/// Print every input device by name. The user picks one with `--device`.
pub fn list_input_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Input devices:");
    for device in host.input_devices()? {
        // `name()` can fail for a disconnected device, so default to a marker.
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        // Report the channel count the app will ACTUALLY use (max-channel pick),
        // not the macOS default, which under-reports for aggregate devices.
        let channels = pick_max_channel_config(&device)
            .map(|c| c.channels().to_string())
            .unwrap_or_else(|_| "?".to_string());
        println!("  - {name}  ({channels} channels)");

        // List every supported config range, so you can see the real options.
        if let Ok(ranges) = device.supported_input_configs() {
            for r in ranges {
                println!(
                    "      {} ch, {}..{} Hz, {:?}",
                    r.channels(),
                    r.min_sample_rate().0,
                    r.max_sample_rate().0,
                    r.sample_format()
                );
            }
        }
    }
    Ok(())
}

/// Find the input device to record from.
///
/// If `wanted` is `Some`, we match the first device whose name *contains* that
/// text (case-insensitive), so "aggregate" matches "My Aggregate Device". If
/// `wanted` is `None`, we use the system default input.
fn pick_device(wanted: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();

    match wanted {
        Some(substr) => {
            let needle = substr.to_lowercase();
            for device in host.input_devices()? {
                if let Ok(name) = device.name() {
                    if name.to_lowercase().contains(&needle) {
                        return Ok(device);
                    }
                }
            }
            Err(anyhow!("no input device name contains {substr:?}"))
        }
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device found")),
    }
}

/// Open the capture stream and start it.
///
/// Returns the live `Stream` plus its config. The caller MUST keep the returned
/// `Stream` alive: when it is dropped, capture stops. Samples flow out through
/// the `sender` you pass in, as `Vec<f32>` batches of interleaved samples.
///
/// Rust note: `cpal::Stream` is not `Send` on macOS, so it cannot move to
/// another thread. That is why we build and hold it on the main thread.
pub fn start_capture(
    wanted: Option<&str>,
    sender: Sender<Vec<f32>>,
) -> Result<(cpal::Stream, CaptureConfig)> {
    let device = pick_device(wanted)?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

    // Choose the configuration with the MOST channels.
    //
    // Why not just `default_input_config()`? For an Aggregate Device, macOS
    // often reports a default of ONE channel (it follows the clock sub-device),
    // even though the aggregate really exposes several. Taking the default would
    // silently drop every channel but one — so we would capture only your mic
    // and never the call output. Picking the max-channel config avoids that.
    let chosen = pick_max_channel_config(&device)?;

    let config = CaptureConfig {
        device_name: name.clone(),
        sample_rate: chosen.sample_rate().0,
        channels: chosen.channels(),
    };

    println!(
        "Recording from: {name}  ({} Hz, {} channel(s), {:?})",
        config.sample_rate,
        config.channels,
        chosen.sample_format()
    );

    // A shared error handler for the audio thread. It only logs; audio errors
    // are rare and we do not want to crash a live recording over one glitch.
    let err_fn = |err| eprintln!("audio stream error: {err}");

    let stream_config: cpal::StreamConfig = chosen.clone().into();

    // The hardware may hand us f32, i16, or u16 samples. We convert every case
    // to f32 (the format the rest of the program expects) inside the callback.
    let stream = match chosen.sample_format() {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &stream_config, sender, err_fn)?,
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &stream_config, sender, err_fn)?,
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &stream_config, sender, err_fn)?,
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    };

    stream.play().context("failed to start audio stream")?;
    Ok((stream, config))
}

/// Pick the supported input config with the most channels. Among equal channel
/// counts, prefer an f32 format, then a sample rate near 48 kHz.
///
/// Falls back to `default_input_config()` if the device reports no supported
/// config ranges, so this never fails where the old code worked.
fn pick_max_channel_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    // Each item is a RANGE (a min/max sample rate for a channel count + format).
    let ranges: Vec<cpal::SupportedStreamConfigRange> = match device.supported_input_configs() {
        Ok(it) => it.collect(),
        Err(_) => Vec::new(),
    };

    if ranges.is_empty() {
        return device
            .default_input_config()
            .context("device has no default input config");
    }

    // Keep only the ranges with the most channels — that is the whole point.
    let max_ch = ranges.iter().map(|r| r.channels()).max().unwrap_or(1);
    let widest: Vec<_> = ranges
        .into_iter()
        .filter(|r| r.channels() == max_ch)
        .collect();

    // Prefer a SANE, standard sample rate. Aggregate devices report absurd
    // rates too (88_200, 176_400, even 768_000). Grabbing the maximum landed on
    // one of those. Instead pick the first standard rate a range supports.
    // 24_000 is here for Bluetooth headsets in hands-free (HFP) mode, e.g. the
    // AirPods mic, which only offer 24/16/8 kHz.
    const PREFERRED: [u32; 5] = [48_000, 44_100, 32_000, 24_000, 16_000];
    for &want in &PREFERRED {
        if let Some(r) = widest
            .iter()
            .find(|r| r.min_sample_rate().0 <= want && want <= r.max_sample_rate().0)
        {
            return Ok(r.clone().with_sample_rate(cpal::SampleRate(want)));
        }
    }

    // No standard rate is available: fall back to the range with the LOWEST
    // rate (avoids the absurd high ones), clamped toward 48 kHz.
    let best = widest
        .iter()
        .min_by_key(|r| r.min_sample_rate().0)
        .expect("widest is non-empty");
    let min = best.min_sample_rate().0;
    let max = best.max_sample_rate().0;
    let target = 48_000u32.clamp(min, max);
    Ok(best.clone().with_sample_rate(cpal::SampleRate(target)))
}

/// Build an input stream for one concrete sample type `T` and forward every
/// buffer as `Vec<f32>` through `sender`.
///
/// Rust note: `T: cpal::SizedSample + cpal::FromSample<...>` is a "trait bound".
/// It means "this function works for any type T that cpal can capture and that
/// we can convert into f32". `dasp_sample`-style `to_sample()` does the convert.
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: Sender<Vec<f32>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32> + 'static,
    f32: cpal::FromSample<T>,
{
    let stream = device.build_input_stream(
        config,
        // This closure is the real-time callback. Keep it tiny.
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Convert the hardware samples to f32 and copy into an owned Vec.
            let batch: Vec<f32> = data.iter().map(|&s| f32::from_sample(s)).collect();

            // Send to the worker. If the worker has hung up (program exiting),
            // `send` errors; we ignore it because there is nothing to do here.
            let _ = sender.send(batch);
        },
        err_fn,
        None, // no timeout
    )?;
    Ok(stream)
}
