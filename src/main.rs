mod infrastructure;
mod domain;

use infrastructure::audio::Segmenter;
use infrastructure::augmentation::Augmenter;
use domain::errors::AppError;
use domain::entities::AudioSegment;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;

fn main() -> Result<(), AppError> {
    let segmenter = Segmenter::new(2.0, 16000, "data/segmented/");
    let total_files = 56;

    if !Path::new("data/segmented/").exists() || fs::read_dir("data/segmented/")?.count() < total_files {
        println!("Running segmentation...");
        segmenter.setup_output_dir()?;
        let pb = ProgressBar::new(total_files as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-")
        );
        pb.set_message("Segmenting audio files");

        for i in 1..=total_files {
            let input_path = format!("data/raw/{:02}.wav", i);
            segmenter.process_file(&input_path, i)?;
            pb.inc(1);
        }
        pb.finish_with_message("Segmentation completed!");
    } else {
        println!("Segmentation already done, skipping...");
    }

    let augmenter = Augmenter::new(16000, "data/augmented/");
    let segmented_dir = fs::read_dir("data/segmented/")?;
    let total_segments = segmented_dir.count();

    if !Path::new("data/augmented/").exists() || fs::read_dir("data/augmented/")?.count() < total_segments * 9 {
        println!("Running augmentation...");
        augmenter.setup_output_dir()?;
        let pb = ProgressBar::new(total_segments as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-")
        );
        pb.set_message("Augmenting audio files");

        for entry in fs::read_dir("data/segmented/")? {
            let entry = entry?;
            let path = entry.path();
            let path_str = path.to_str().ok_or_else(|| AppError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid path"
            )))?;
            
            let audio_data = dasp_rs::audio_io::wav::load(path_str, None, Some(true), None, None)?;
            let segment = AudioSegment::new(audio_data.samples, audio_data.sample_rate);
            
            augmenter.augment_file(&segment, path_str)?;
            pb.inc(1);
        }
        pb.finish_with_message("Augmentation completed!");
    } else {
        println!("Augmentation already done, skipping...");
    }

    Ok(())
}