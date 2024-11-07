use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample;
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::TAU;

const IN_TRIGGER_ADB: f32 = 34.0;
const PLAYBACK_MAX_ADB: f32 = 30.0;
const MIN_CLIP_SECS: f32 = 0.25;

const SIN_OSC_FREQ: f32 = 20.0;
const SIN_OSC_AMP: f32 = 0.04;

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();

    let input = host
        .default_input_device()
        .expect("No input device available");
    println!(
        "Using output device {}",
        input.name().unwrap_or("unnamed".to_owned())
    );
    let input_config = input
        .default_input_config()
        .expect("No default input config");
    let input_sample_format = input_config.sample_format();
    let input_sample_rate = input_config.sample_rate().0 as f32;
    let input_channels = input_config.channels() as usize;

    let output = host
        .default_output_device()
        .expect("No output device available");
    println!(
        "Using output device {}",
        output.name().unwrap_or("unnamed".to_owned())
    );
    let output_config = output
        .default_output_config()
        .expect("No default output config");
    let output_sample_format = output_config.sample_format();
    let output_sample_rate = output_config.sample_rate().0 as f32;
    let output_channels = output_config.channels() as usize;

    assert_eq!(input_sample_format, output_sample_format);
    assert_eq!(input_sample_rate, output_sample_rate);
    assert_eq!(input_channels, output_channels);

    let config: cpal::StreamConfig = output_config.into();
    // Create a delay in case the i and output devices aren't synced.
    let latency_frames = 1.0 * config.sample_rate.0 as f32;
    let latency_samples = latency_frames as usize * output_channels as usize;

    // The buffer to share samples
    let ring = HeapRb::<f32>::new(latency_samples * 2);
    let (mut producer, mut consumer) = ring.split();

    // Fill the samples with 0.0 equal to the length of the delay.
    for _ in 0..latency_samples {
        // The ring buffer has twice as much space as necessary to add latency here,
        // so this should never fail
        producer.try_push(0.0).unwrap();
    }

    let mut sin_osc = SineOscillator::new(SIN_OSC_AMP, SIN_OSC_FREQ, output_sample_rate);
    let mut clipper = LoudClipCollector::new(IN_TRIGGER_ADB, MIN_CLIP_SECS, output_sample_rate);

    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut output_fell_behind = false;

        let mut samples: Vec<f32> = vec![];
        samples.clone_from_slice(data);
        clipper.ingest(Chunk::new(samples, input_sample_rate));

        match clipper.next() {
            Some(samples) => {
                for sample in samples {
                    if producer.try_push(sample).is_err() {
                        output_fell_behind = true;
                    }
                }
            }
            None => {
                for i in 0..data.len() {
                    if producer.try_push(sin_osc.next().unwrap()).is_err() {
                        output_fell_behind = true;
                    }
                }
            }
        };
        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let mut input_fell_behind = false;
        let decibels = calculate_dba(&data, input_sample_rate);
        println!("Max decibels for chunk: {}", decibels);

        for sample in data {
            *sample = match consumer.try_pop() {
                Some(s) => s,
                None => {
                    input_fell_behind = true;
                    0.0
                }
            };
        }
        if input_fell_behind {
            eprintln!("input stream fell behind: try increasing latency");
        }
    };

    // Build streams.
    println!(
        "Attempting to build both streams with f32 samples and `{:?}`.",
        config
    );
    let input_stream = input.build_input_stream(&config, input_data_fn, err_fn, None)?;
    let output_stream = output.build_output_stream(&config, output_data_fn, err_fn, None)?;
    println!("Successfully built streams.");

    input_stream.play()?;
    output_stream.play()?;

    // Keep the program running until the input stream is paused
    std::thread::sleep(std::time::Duration::from_secs(60 * 60));

    Ok(())
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}

fn calculate_dba(samples: &[f32], sample_rate: f32) -> f32 {
    let len = samples.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(len);

    // Convert samples to complex numbers for FFT
    let mut complex_samples: Vec<Complex<f32>> =
        samples.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Perform FFT
    fft.process(&mut complex_samples);

    // Apply A-weighting
    let freqs: Vec<f32> = (0..len)
        .map(|i| (i as f32 * sample_rate as f32) / len as f32)
        .collect();
    for (i, freq) in freqs.iter().enumerate() {
        let a_weight = a_weighting(*freq);
        complex_samples[i] = complex_samples[i] * a_weight;
    }

    // Perform IFFT (inverse FFT) to get back to time domain
    let ifft = planner.plan_fft_inverse(len);
    ifft.process(&mut complex_samples);

    // Compute RMS of the A-weighted signal
    let rms = complex_samples.iter().map(|c| c.norm_sqr()).sum::<f32>() / len as f32;
    let rms = rms.sqrt();

    // Convert RMS to decibels
    20.0 * rms.log10()
}

// Helper function for A-weighting calculation
fn a_weighting(frequency: f32) -> f32 {
    // A-weighting formula constants
    let c1 = 12200.0_f32.powi(2);
    let c2 = 20.6_f32.powi(2);
    let c3 = 107.7_f32.powi(2);
    let c4 = 737.9_f32.powi(2);
    let f2 = frequency.powi(2);

    let a = (c1 * f2 * f2) / ((f2 + c2) * (f2 + c3).sqrt() * (f2 + c4).sqrt() * (f2 + c1));
    a.sqrt()
}

pub struct SineOscillator {
    amplitude: f32,
    frequency: f32,
    sample_rate: f32,
    phase: f32,
    phase_increment: f32,
}

impl SineOscillator {
    /// Creates a new sine oscillator with a given amplitude, frequency, and sample rate
    pub fn new(amplitude: f32, frequency: f32, sample_rate: f32) -> Self {
        let phase_increment = frequency / sample_rate * TAU;
        Self {
            amplitude,
            frequency,
            sample_rate,
            phase: 0.0,
            phase_increment,
        }
    }

    /// Sets a new amplitude for the oscillator
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Resets the phase of the oscillator
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }
}

impl Iterator for SineOscillator {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.amplitude * (self.phase).sin();
        self.phase += self.phase_increment;
        if self.phase >= TAU {
            self.phase -= TAU; // Wrap phase to stay within 0 to TAU
        }
        Some(sample)
    }
}

pub struct Chunk {
    decibels: f32,
    samples: Vec<f32>,
}

impl Chunk {
    pub fn new(samples: Vec<f32>, sample_rate: f32) -> Chunk {
        let decibels = calculate_dba(samples.as_slice(), sample_rate);
        Chunk { decibels, samples }
    }
}

pub struct LoudClipCollector {
    buffer: Vec<Chunk>,
    playback: bool,
    sample_rate: f32,
    min_clip_secs: f32,
    loud_gate_adb: f32,
}

impl LoudClipCollector {
    pub fn new(loud_gate_adb: f32, min_clip_secs: f32, sample_rate: f32) -> Self {
        let mut buffer: Vec<Chunk> = vec![];
        let playback: bool = false;
        return LoudClipCollector {
            buffer,
            playback,
            sample_rate,
            min_clip_secs,
            loud_gate_adb,
        };
    }

    pub fn ingest(mut self, chunk: Chunk) {
        self.buffer.insert(0, chunk);
        let avg_loudness: f32 =
            self.buffer.iter().map(|chunk| chunk.decibels).sum::<f32>() / self.buffer.len() as f32;

        if avg_loudness < self.loud_gate_adb {
            let duration = self.sample_rate
                / self
                    .buffer
                    .iter()
                    .map(|chunk| chunk.samples.len() as f32)
                    .sum::<f32>();
            if duration >= self.min_clip_secs {
                self.playback = true;
            } else {
                self.buffer.clear();
            }
        }
    }
}

impl Iterator for LoudClipCollector {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.playback {
            match self.buffer.pop() {
                Some(chunk) => return Some(chunk.samples),
                None => self.playback = false,
            }
        }
        None
    }
}
