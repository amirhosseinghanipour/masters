use crate::domain::entities::AudioSegment;
use crate::domain::errors::AppError;
use dasp_rs::audio_io::wav::{load, export_to_wav};
use dasp_rs::signal_processing::resampling::resample;
use std::fs;

pub struct Segmenter {
    segment_duration: f64, 
    target_sample_rate: u32, 
    output_dir: String,
}

impl Segmenter {
    pub fn new(segment_duration: f64, target_sample_rate: u32, output_dir: impl Into<String>) -> Self {
        Segmenter {
            segment_duration,
            target_sample_rate,
            output_dir: output_dir.into(),
        }
    }

    pub fn process_file(&self, path: &str, file_id: usize) -> Result<(), AppError> {
        let segments = self.segment_file(path)?;
        self.save_segments(&segments, file_id)?;
        Ok(())
    }

    fn segment_file(&self, path: &str) -> Result<Vec<AudioSegment>, AppError> {
        let audio_data = load(path, None, Some(true), None, None)?; 
        let original_rate = audio_data.sample_rate;
        let resampled_samples = if original_rate != self.target_sample_rate {
            resample(&audio_data.samples, original_rate, self.target_sample_rate)?
        } else {
            audio_data.samples
        };

        let segment_length = (self.segment_duration * self.target_sample_rate as f64) as usize;
        let mut segments = Vec::new();

        for chunk in resampled_samples.chunks(segment_length) {
            if chunk.len() == segment_length {
                segments.push(AudioSegment::new(chunk.to_vec(), self.target_sample_rate));
            }
        }

        Ok(segments)
    }

    fn save_segments(&self, segments: &[AudioSegment], file_id: usize) -> Result<(), AppError> {
        for (i, segment) in segments.iter().enumerate() {
            let output_path = format!("{}/{:02}_{:03}.wav", self.output_dir, file_id, i);
            let audio_data = dasp_rs::AudioData {
                samples: segment.samples.clone(),
                sample_rate: segment.sample_rate,
                channels: 1,
            };
            export_to_wav(&output_path, &audio_data)?;
        }
        Ok(())
    }

    pub fn setup_output_dir(&self) -> Result<(), AppError> {
        fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }
}