use candle_core::{Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::distance_functions::cosine_similarity::CosineSimilarity;
use neural_networks::layers::param_set::ParamSet;
use neural_networks::{container::sequential::Sequential, loss_functions::mse_loss::MSELoss};
use std::collections::HashMap;
use std::io::Write;
use utils::derivative::DerivativeMorphism;
use utils::morphism::Morphism;

use utils::errors::{CategoryError, CategoryResult};

pub fn add_all(a: &[ParamSet], b: &[ParamSet]) -> CategoryResult<Vec<ParamSet>> {
    if a.len() != b.len() {
        return Err(CategoryError::InvalidInput(format!(
            "layer count mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.add(y).map_err(CategoryError::from))
        .collect()
}

pub fn sgd_all(
    params: &[ParamSet],
    grads: &[ParamSet],
    learning_rate: f64,
) -> CategoryResult<Vec<ParamSet>> {
    if params.len() != grads.len() {
        return Err(CategoryError::InvalidInput(format!(
            "params/grads layer count mismatch: {} vs {}",
            params.len(),
            grads.len()
        )));
    }

    params
        .iter()
        .zip(grads.iter())
        .map(|(param, grad)| param.sgd(grad, learning_rate).map_err(CategoryError::from))
        .collect()
}
pub struct SiameseTrainStep {
    dataset: DataSet,
    learning_rate: f64,
    batch_size: usize,
    cut_shape: (usize, usize),
    device: Device,
}

impl SiameseTrainStep {
    pub fn export_embeddings_2d(
        &self,
        embed: &mut Sequential,
        dataset: &DataSet,
        limit: usize,
        out_path: &str,
    ) -> CategoryResult<()> {
        let mut all_embs: Vec<Vec<f32>> = Vec::new();
        let mut all_labels: Vec<u8> = Vec::new();

        for batch in dataset.batch_view_iter(200, limit) {
            let images = batch.images_tensor(&self.device)?;
            let emb = embed.forward(images)?; // (batch, dim)
            let emb_vec: Vec<Vec<f32>> = emb.to_vec2()?;
            all_embs.extend(emb_vec);
            all_labels.extend(batch.labels.iter().copied());
        }

        let points_2d = pca_2d(&all_embs);

        let mut file = std::fs::File::create(out_path).expect("สร้างไฟล์ output ไม่ได้");
        writeln!(file, "x,y,label").ok();
        for ((x, y), label) in points_2d.iter().zip(all_labels.iter()) {
            writeln!(file, "{},{},{}", x, y, label).ok();
        }

        Ok(())
    }
    pub fn new(
        dataset: DataSet,
        learning_rate: f64,
        batch_size: usize,
        cut_shape: (usize, usize),
        device: Device,
    ) -> Self {
        Self {
            dataset,
            learning_rate,
            batch_size,
            cut_shape,
            device,
        }
    }

    pub fn train(&self, embed: &mut Sequential) -> CategoryResult<()> {
        let loss_fn = MSELoss;
        let cos_fn = CosineSimilarity::default();
        let steps_per_epoch = (self.dataset.labels.len() + self.batch_size - 1) / self.batch_size;
        for epoch in 0..10 {
            for step in 0..steps_per_epoch {
                let pair =
                    self.dataset
                        .sample_pairs(self.batch_size, self.cut_shape, &self.device)?;
                let (left_emb, left_tape) = embed.forward_with_tape(pair.left)?;
                let (right_emb, right_tape) = embed.forward_with_tape(pair.right)?;
                let target = pair.labels.to_dtype(left_emb.dtype())?.affine(2.0, -1.0)?;
                let similarity = cos_fn.apply((left_emb.clone(), right_emb.clone()))?;
                let loss = loss_fn.apply((similarity.clone(), target.clone()))?;
                let d_loss_d_similarity = loss_fn.derivative(()).apply((similarity, target))?;

                let (d_cos_d_left, d_cos_d_right) =
                    cos_fn.derivative(()).apply((left_emb, right_emb))?;
                let upstream = d_loss_d_similarity.unsqueeze(1)?;
                let delta_left = d_cos_d_left.broadcast_mul(&upstream)?;
                let delta_right = d_cos_d_right.broadcast_mul(&upstream)?;
                let grads_left = embed.backward_with_tape(left_tape, delta_left)?;
                let grads_right = embed.backward_with_tape(right_tape, delta_right)?;

                let grads = add_all(&grads_left, &grads_right)?;
                let new_params = sgd_all(embed.params(), &grads, self.learning_rate)?;
                embed.set_params(new_params);

                if step % 100 == 0 {
                    println!("epoch: {}/10, step: {}, loss: {}", epoch + 1, step, loss);
                }
            }
        }
        Ok(())
    }
    fn compute_class_centroids(
        &self,
        embed: &mut Sequential,
    ) -> CategoryResult<HashMap<u8, Tensor>> {
        let mut sums: HashMap<u8, Tensor> = HashMap::new();

        let mut counts: HashMap<u8, usize> = HashMap::new();

        for batch in self.dataset.batch_view_iter(100, self.dataset.labels.len()) {
            let images = batch.images_tensor(&self.device)?;

            let emb = embed.forward(images)?;
            let emb = l2_normalize_rows(&emb)?;

            for (row_idx, &label) in batch.labels.iter().enumerate() {
                // shape: [1, embedding_dim]
                let row = emb.narrow(0, row_idx, 1)?;

                if let Some(current_sum) = sums.get(&label) {
                    let next_sum = (current_sum + &row)?;

                    sums.insert(label, next_sum);
                } else {
                    sums.insert(label, row);
                }

                *counts.entry(label).or_insert(0) += 1;
            }
        }

        let mut centroids = HashMap::with_capacity(sums.len());

        for (label, sum) in sums {
            let count = *counts.get(&label).ok_or_else(|| {
                CategoryError::InvalidInput(format!("missing count for label {}", label))
            })?;

            let mean = sum.affine(1.0 / count as f64, 0.0)?;

            // normalize centroid อีกครั้ง
            let centroid = l2_normalize_rows(&mean)?;

            centroids.insert(label, centroid);
        }

        Ok(centroids)
    }

    pub fn test(&self, embed: &mut Sequential, testset: DataSet) -> CategoryResult<()> {
        let centroids = self.compute_class_centroids(embed)?;

        let cos_fn = CosineSimilarity::default();

        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for (batch_idx, batch) in testset.batch_view_iter(100, 10_000).enumerate() {
            let images = batch.images_tensor(&self.device)?;

            let emb = embed.forward(images)?;

            // normalize test embedding ให้เหมือนกับ centroid
            let emb = l2_normalize_rows(&emb)?;

            let batch_size = emb.dim(0)?;
            let mut correct_in_batch = 0usize;

            for row_idx in 0..batch_size {
                let sample = emb.narrow(0, row_idx, 1)?;

                let mut best_label: Option<u8> = None;

                let mut best_similarity = f32::NEG_INFINITY;

                for (&label, centroid) in centroids.iter() {
                    let similarity = cos_fn.apply((sample.clone(), centroid.clone()))?;

                    let score = similarity.reshape(())?.to_scalar::<f32>()?;

                    if score > best_similarity {
                        best_similarity = score;
                        best_label = Some(label);
                    }
                }

                if best_label == Some(batch.labels[row_idx]) {
                    correct_in_batch += 1;
                }
            }

            total_correct += correct_in_batch;
            total_samples += batch_size;

            println!(
                "batch: {}, correct: {}/{}",
                batch_idx, correct_in_batch, batch_size,
            );
        }

        if total_samples == 0 {
            return Err(CategoryError::InvalidInput("testset is empty".to_string()));
        }

        let accuracy = total_correct as f64 / total_samples as f64 * 100.0;

        println!(
            "Accuracy = {:.2}% ({}/{})",
            accuracy, total_correct, total_samples,
        );

        Ok(())
    }
}
fn pca_2d(data: &[Vec<f32>]) -> Vec<(f32, f32)> {
    let n = data.len();
    let dim = data[0].len();

    let mut mean = vec![0.0f64; dim];
    for row in data {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v as f64;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .map(|(&v, &m)| v as f64 - m)
                .collect()
        })
        .collect();

    // covariance matrix (dim x dim) = X^T X / n
    let mut cov = vec![vec![0.0f64; dim]; dim];
    for row in &centered {
        for i in 0..dim {
            if row[i] == 0.0 {
                continue;
            }
            for j in i..dim {
                cov[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..dim {
        for j in i..dim {
            cov[i][j] /= n as f64;
            cov[j][i] = cov[i][j];
        }
    }

    let pc1 = power_iteration(&cov, dim, 100);
    let eigval1 = rayleigh_quotient(&cov, &pc1);

    let mut cov_deflated = cov.clone();
    for i in 0..dim {
        for j in 0..dim {
            cov_deflated[i][j] -= eigval1 * pc1[i] * pc1[j];
        }
    }
    let pc2 = power_iteration(&cov_deflated, dim, 100);

    centered
        .iter()
        .map(|row| {
            let x: f64 = row.iter().zip(pc1.iter()).map(|(&a, &b)| a * b).sum();
            let y: f64 = row.iter().zip(pc2.iter()).map(|(&a, &b)| a * b).sum();
            (x as f32, y as f32)
        })
        .collect()
}

fn power_iteration(mat: &[Vec<f64>], dim: usize, iters: usize) -> Vec<f64> {
    let mut v = vec![1.0f64 / (dim as f64).sqrt(); dim];
    for _ in 0..iters {
        let mut new_v = vec![0.0f64; dim];
        for i in 0..dim {
            for j in 0..dim {
                new_v[i] += mat[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for x in new_v.iter_mut() {
            *x /= norm;
        }
        v = new_v;
    }
    v
}

fn rayleigh_quotient(mat: &[Vec<f64>], v: &[f64]) -> f64 {
    let dim = v.len();
    let mut mv = vec![0.0f64; dim];
    for i in 0..dim {
        for j in 0..dim {
            mv[i] += mat[i][j] * v[j];
        }
    }
    mv.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}
fn l2_normalize_rows(x: &Tensor) -> CategoryResult<Tensor> {
    // x: [batch_size, embedding_dim]
    let eps = 1e-8_f64;

    let norm = x.sqr()?.sum(1)?.sqrt()?.affine(1.0, eps)?;

    Ok(x.broadcast_div(&norm.unsqueeze(1)?)?)
}
