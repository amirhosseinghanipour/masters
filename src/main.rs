mod infrastructure;
mod domain;

use infrastructure::audio::Segmenter;
use domain::errors::AppError;
use indicatif::{ProgressBar, ProgressStyle};

fn main() -> Result<(), AppError> {
    let segmenter = Segmenter::new(2.0, 16000, "data/segmented/");
    segmenter.setup_output_dir()?;

    let total_files = 56;
    let pb = ProgressBar::new(total_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("Segmenting audio files");

    for i in 1..=total_files {
        let input_path = format!("data/raw/{:02}.wav", i);
        segmenter.process_file(&input_path, i)?;
        pb.inc(1);
    }

    pb.finish_with_message("Segmentation completed successfully!");
    Ok(())
}