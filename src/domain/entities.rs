#[derive(Debug)]
pub struct AudioSegment {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioSegment {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        AudioSegment { samples, sample_rate }
    }
}