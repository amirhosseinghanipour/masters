use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Audio processing error: {0}")]
    Audio(#[from] dasp_rs::AudioError),
}