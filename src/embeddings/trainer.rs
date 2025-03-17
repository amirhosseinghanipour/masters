use crate::domain::entities::AudioSegment;
use crate::domain::errors::AppError;
use tch::{nn, Tensor, Device, Kind, TchError, CModule};
use std::path::Path;
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use toml;

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    training: TrainingConfig,
    model: ModelConfig,
    evaluation: EvaluationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingConfig {
    batch_size: usize,
    epochs: usize,
    temperature: f32,
    data_dir: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelConfig {
    device: String,
    embedding_size: i64,
    model_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluationConfig {
    output_file: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResult {
    speaker_id: String,
    filename: String,
    embedding: Vec<f32>,
}

pub struct Trainer {
    model: WavLMModel,
    projection_head: nn::Sequential,
    device: Device,
    config: Config,
    var_store: nn::VarStore,
}

pub struct WavLMModel {
    module: CModule,
}

impl WavLMModel {
    pub fn new(device: Device, model_path: &str) -> Result<Self, AppError> {
        let module = CModule::load_on_device(model_path, device)?;
        Ok(WavLMModel { module })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, AppError> {
        let output = self.module.forward_ts(&[input.shallow_clone()])?;
        Ok(output)
    }
}

impl Trainer {
    pub fn new(config_path: &str) -> Result<Self, AppError> {
        let config_str = fs::read_to_string(config_path)?;
        let config: Config = toml::from_str(&config_str)?;

        let device = match config.model.device.as_str() {
            "cuda_if_available" => Device::cuda_if_available(),
            "cuda:0" => Device::Cuda(0),
            "cpu" => Device::Cpu,
            _ => return Err(AppError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid device specification in config",
            ))),
        };

        let model = WavLMModel::new(device, &config.model.model_path)?;
        let mut var_store = nn::VarStore::new(device);
        let p = &var_store.root();

        let projection_head = nn::seq()
            .add(nn::linear(p / "layer1", 768, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "layer2", 512, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "output", 256, config.model.embedding_size, Default::default()))
            .add_fn(|xs| xs.l2_normalize(-1));

        Ok(Trainer {
            model,
            projection_head,
            device,
            config,
            var_store,
        })
    }

    fn nt_xent_loss(&self, embeddings: &Tensor) -> Result<Tensor, TchError> {
        let similarity_matrix = embeddings.matmul(&embeddings.transpose(-1, -2)) / self.config.training.temperature;
        let labels = Tensor::arange(embeddings.size()[0] as i64, (Kind::Int64, self.device));
        tch::vision::cross_entropy_for_logits(&similarity_matrix, &labels)
    }

    fn group_files_by_speaker(&self) -> Result<HashMap<String, Vec<String>>, AppError> {
        let mut speaker_files: HashMap<String, Vec<String>> = HashMap::new();
        for entry in fs::read_dir(&self.config.training.data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                let parts: Vec<&str> = filename.split('_').collect();
                if parts.len() >= 2 {
                    let speaker_id = parts[0].to_string();
                    speaker_files
                        .entry(speaker_id)
                        .or_insert_with(Vec::new)
                        .push(path.to_str().unwrap().to_string());
                }
            }
        }
        Ok(speaker_files)
    }

    fn sample_pairs(&self, speaker_files: &HashMap<String, Vec<String>>) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let speaker_ids: Vec<&String> = speaker_files.keys().collect();

        for _ in 0..self.config.training.batch_size / 2 {
            if let Some(speaker_id) = speaker_ids.choose(&mut thread_rng()) {
                if let Some(files) = speaker_files.get(*speaker_id) {
                    if files.len() >= 2 {
                        let mut rng = thread_rng();
                        let pair: Vec<&String> = files.choose_multiple(&mut rng, 2).collect();
                        pairs.push((pair[0].clone(), pair[1].clone()));
                    }
                }
            }
        }
        pairs
    }

    pub fn train(&self) -> Result<(), AppError> {
        let opt = nn::Adam::default().build(&self.var_store, 1e-5)?;
        let speaker_files = self.group_files_by_speaker()?;
        let total_steps = (speaker_files.len() * self.config.training.epochs) as u64;

        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
                .progress_chars("#>-")
        );
        pb.set_message("Training model");

        for epoch in 0..self.config.training.epochs {
            let mut all_files: Vec<String> = speaker_files.values().flatten().cloned().collect();
            all_files.shuffle(&mut thread_rng());

            for _ in 0..(total_steps / self.config.training.epochs as u64) {
                let pairs = self.sample_pairs(&speaker_files);
                let mut batch_tensors = Vec::new();

                for (file1, file2) in pairs {
                    for file in &[file1, file2] {
                        let audio_data = dasp_rs::audio_io::wav::load(file, None, Some(true), None, None)?;
                        let segment = AudioSegment::new(audio_data.samples, audio_data.sample_rate);
                        let tensor = Tensor::of_slice(&segment.samples)
                            .to_device(self.device)
                            .view([1, -1]);
                        batch_tensors.push(tensor);
                    }
                }

                let batch = Tensor::cat(&batch_tensors, 0);
                let embeddings = tch::no_grad(|| self.model.forward(&batch))?;
                let projected = self.projection_head.forward(&embeddings);
                let loss = self.nt_xent_loss(&projected)?;
                
                opt.backward_step(&loss)?;

                pb.inc(1);
            }
            println!("Epoch {} completed", epoch + 1);
        }
        pb.finish_with_message("Training completed!");
        Ok(())
    }

    pub fn evaluate(&self) -> Result<(), AppError> {
        let speaker_files = self.group_files_by_speaker()?;
        let mut results = Vec::new();

        let pb = ProgressBar::new(speaker_files.values().map(|v| v.len() as u64).sum());
        pb.set_message("Extracting embeddings");

        for (speaker_id, files) in speaker_files {
            for file in files {
                let audio_data = dasp_rs::audio_io::wav::load(&file, None, Some(true), None, None)?;
                let segment = AudioSegment::new(audio_data.samples, audio_data.sample_rate);
                let embedding = self.get_embeddings(&segment)?;
                let embedding_vec: Vec<f32> = embedding
                    .to_device(Device::Cpu)
                    .into_shape([self.config.model.embedding_size])?
                    .into();

                results.push(EmbeddingResult {
                    speaker_id: speaker_id.clone(),
                    filename: Path::new(&file).file_name().unwrap().to_str().unwrap().to_string(),
                    embedding: embedding_vec,
                });
                pb.inc(1);
            }
        }

        let json = serde_json::to_string_pretty(&results)?;
        let mut file = File::create(&self.config.evaluation.output_file)?;
        file.write_all(json.as_bytes())?;
        pb.finish_with_message("Embeddings saved!");
        Ok(())
    }

    pub fn get_embeddings(&self, segment: &AudioSegment) -> Result<Tensor, AppError> {
        let input = Tensor::of_slice(&segment.samples)
            .to_device(self.device)
            .view([1, -1]);
        let embeddings = tch::no_grad(|| self.model.forward(&input))?;
        Ok(self.projection_head.forward(&embeddings))
    }
}