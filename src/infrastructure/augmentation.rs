use crate::domain::entities::AudioSegment;
use crate::domain::errors::AppError;
use dasp_rs::audio_io::wav::export_to_wav;
use rand::Rng;
use rustfft::FftPlanner;
use num_complex::Complex;
use std::fs;
use std::path::Path;

#[allow(dead_code)]
pub trait Augmentation {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError>;
    fn name(&self) -> &'static str;
}

#[allow(dead_code)]
pub struct Augmenter<'a> {
    sample_rate: u32,
    output_dir: &'a str,
    augmentations: Vec<Box<dyn Augmentation>>,
}

#[allow(dead_code)]
impl<'a> Augmenter<'a> {
    pub fn new(sample_rate: u32, output_dir: &'a str) -> Self {
        let augmentations: Vec<Box<dyn Augmentation>> = vec![
            Box::new(BackgroundNoise { level: 0.02 }),
            Box::new(TimeStretch { rate: 1.5, target_length: sample_rate * 2 }),
            Box::new(TimeStretch { rate: 0.5, target_length: sample_rate * 2 }),
            Box::new(PitchShift { n_steps: 3.0 }),
            Box::new(PitchShift { n_steps: -3.0 }),
            Box::new(TimeShift { shift_amount: 500 }),
            Box::new(VolumeScale { factor: 1.5 }),
            Box::new(VolumeScale { factor: 0.5 }),
            Box::new(ClipDistortion { level: 0.4 }),
        ];
        Augmenter { sample_rate, output_dir, augmentations }
    }

    pub fn setup_output_dir(&self) -> Result<(), AppError> {
        fs::create_dir_all(self.output_dir)?;
        Ok(())
    }

    pub fn augment_file(&self, segment: &AudioSegment, original_path: &str) -> Result<(), AppError> {
        let path = Path::new(original_path);
        let base_name = path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AppError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid filename"
            )))?;

        for aug in &self.augmentations {
            let augmented = aug.apply(segment)?;
            let output_path = format!("{}/{}_{}.wav", self.output_dir, base_name, aug.name());
            let audio_data = dasp_rs::AudioData {
                samples: augmented.samples.clone(),
                sample_rate: augmented.sample_rate,
                channels: 1,
            };
            export_to_wav(&output_path, &audio_data)?;
        }
        Ok(())
    }
}

struct BackgroundNoise {
    level: f32,
}

impl Augmentation for BackgroundNoise {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let mut rng = rand::rng();
        let noisy_samples: Vec<f32> = segment.samples.iter()
            .map(|&s| s + rng.random_range(-self.level..self.level))
            .collect();
        Ok(AudioSegment::new(noisy_samples, segment.sample_rate))
    }

    fn name(&self) -> &'static str { "noise" }
}

struct TimeStretch {
    rate: f32,
    target_length: u32,
}

impl Augmentation for TimeStretch {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let new_length = (segment.samples.len() as f32 / self.rate) as usize;
        let mut stretched = Vec::with_capacity(new_length);
        for i in 0..new_length {
            let orig_idx = i as f32 * self.rate;
            let idx_floor = orig_idx.floor() as usize;
            let frac = orig_idx - idx_floor as f32;
            if idx_floor + 1 < segment.samples.len() {
                let sample = segment.samples[idx_floor] * (1.0 - frac) + segment.samples[idx_floor + 1] * frac;
                stretched.push(sample);
            } else if idx_floor < segment.samples.len() {
                stretched.push(segment.samples[idx_floor]);
            }
        }

        let target = self.target_length as usize;
        if stretched.len() > target {
            stretched.truncate(target);
        } else {
            stretched.extend(vec![0.0; target - stretched.len()]);
        }

        Ok(AudioSegment::new(stretched, segment.sample_rate))
    }

    fn name(&self) -> &'static str {
        if self.rate > 1.0 { "stretch_fast" } else { "stretch_slow" }
    }
}

struct PitchShift {
    n_steps: f32,
}

impl Augmentation for PitchShift {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let factor = 2.0_f32.powf(self.n_steps / 12.0);
        let window_size = 1024;
        let hop_size = window_size / 4;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);
        let ifft = planner.plan_fft_inverse(window_size);

        let mut output = Vec::new();
        let mut buffer = vec![Complex::new(0.0, 0.0); window_size];
        let mut phase_acc = vec![0.0; window_size];
        let mut pos = 0;

        while pos + window_size <= segment.samples.len() {
            buffer.iter_mut().enumerate().for_each(|(i, b)| {
                let hann = 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (window_size as f32 - 1.0)).cos());
                *b = Complex::new(segment.samples[pos + i] * hann, 0.0);
            });

            fft.process(&mut buffer);
            let mag_phase: Vec<(f32, f32)> = buffer.iter().map(|c| (c.norm(), c.arg())).collect();
            let shifted: Vec<Complex<f32>> = mag_phase.iter().enumerate().map(|(i, &(mag, phase))| {
                let new_idx = (i as f32 * factor) as usize;
                if new_idx < window_size {
                    Complex::from_polar(mag, phase + phase_acc[i])
                } else {
                    Complex::new(0.0, 0.0)
                }
            }).collect();

            buffer.copy_from_slice(&shifted);
            ifft.process(&mut buffer);
            buffer.iter_mut().for_each(|c| *c = *c / window_size as f32);

            for (i, &sample) in buffer.iter().enumerate() {
                if pos + i < output.len() {
                    output[pos + i] += sample.re;
                } else {
                    output.push(sample.re);
                }
            }

            pos += hop_size;
            phase_acc = mag_phase.into_iter().map(|(_, p)| p).collect();
        }

        while output.len() < segment.samples.len() {
            output.push(0.0);
        }
        output.truncate(segment.samples.len());

        Ok(AudioSegment::new(output, segment.sample_rate))
    }

    fn name(&self) -> &'static str {
        if self.n_steps > 0.0 { "pitch_up" } else { "pitch_down" }
    }
}

struct TimeShift {
    shift_amount: i32,
}

impl Augmentation for TimeShift {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let mut shifted = segment.samples.clone();
        shifted.rotate_right(self.shift_amount as usize % segment.samples.len());
        Ok(AudioSegment::new(shifted, segment.sample_rate))
    }

    fn name(&self) -> &'static str { "time_shift" }
}

struct VolumeScale {
    factor: f32,
}

impl Augmentation for VolumeScale {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let scaled: Vec<f32> = segment.samples.iter().map(|&s| s * self.factor).collect();
        Ok(AudioSegment::new(scaled, segment.sample_rate))
    }

    fn name(&self) -> &'static str {
        if self.factor > 1.0 { "vol_up" } else { "vol_down" }
    }
}

struct ClipDistortion {
    level: f32,
}

impl Augmentation for ClipDistortion {
    fn apply(&self, segment: &AudioSegment) -> Result<AudioSegment, AppError> {
        let clipped: Vec<f32> = segment.samples.iter().map(|&s| s.clamp(-self.level, self.level)).collect();
        Ok(AudioSegment::new(clipped, segment.sample_rate))
    }

    fn name(&self) -> &'static str { "clip" }
}