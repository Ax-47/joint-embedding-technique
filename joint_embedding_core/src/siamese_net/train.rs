use candle_core::{Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::{container::sequential::Sequential, layers::linear::LinearLayerParams};
use std::collections::HashMap;
use std::io::Write;

use utils::errors::CategoryResult;

fn sgd_step(
    params: &[LinearLayerParams],
    grads: &[LinearLayerParams],
    lr: f64,
) -> CategoryResult<Vec<LinearLayerParams>> {
    let mut new_params = Vec::with_capacity(params.len());
    for (param, grad) in params.iter().zip(grads) {
        new_params.push(LinearLayerParams {
            weight_matrix: (&param.weight_matrix - (lr * &grad.weight_matrix)?)?,
            bias_matrix: (&param.bias_matrix - (lr * &grad.bias_matrix)?)?,
        });
    }
    Ok(new_params)
}

fn add_params(
    a: &[LinearLayerParams],
    b: &[LinearLayerParams],
) -> CategoryResult<Vec<LinearLayerParams>> {
    let mut summed = Vec::with_capacity(a.len());
    for (pa, pb) in a.iter().zip(b) {
        summed.push(LinearLayerParams {
            weight_matrix: (&pa.weight_matrix + &pb.weight_matrix)?,
            bias_matrix: (&pa.bias_matrix + &pb.bias_matrix)?,
        });
    }
    Ok(summed)
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
        embed: &Sequential,
        dataset: &DataSet,
        limit: usize,
        out_path: &str,
    ) -> CategoryResult<()> {
        let mut all_embs: Vec<Vec<f32>> = Vec::new();
        let mut all_labels: Vec<u8> = Vec::new();

        for batch in dataset.batch_view_iter(200, limit) {
            let images = batch.images_tensor(&self.device)?;
            let (emb, _) = embed.forward_with_tape(images)?; // (batch, dim)
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
        let steps_per_epoch = self.dataset.labels.len() / self.batch_size;

        for epoch in 0..10 {
            for step in 0..steps_per_epoch {
                let pair =
                    self.dataset
                        .sample_pairs(self.batch_size, self.cut_shape, &self.device)?;
                embed.currying();
                let (left_emb, left_outs) = embed.forward_with_tape(pair.left)?;

                embed.currying();
                let (right_emb, right_outs) = embed.forward_with_tape(pair.right)?;

                let diff = (&left_emb - &right_emb)?;
                let dist_sq = diff.sqr()?.sum_keepdim(1)?;
                let per_sample_loss = dist_sq.clone();
                let loss = per_sample_loss.mean_all()?;

                let scale = 2.0 / self.batch_size as f64;
                let d_diff = diff.affine(scale, 0.0)?;
                let delta_left = d_diff.clone();
                let delta_right = d_diff.neg()?;

                let grads_left = embed.backward(left_outs, delta_left)?;
                let grads_right = embed.backward(right_outs, delta_right)?;
                let grads = add_params(&grads_left, &grads_right)?;

                let new_params = sgd_step(&embed.params(), &grads, self.learning_rate)?;
                embed.set_params(new_params);

                if step % 100 == 0 {
                    println!("epoch: {}/10, step: {}, loss: {}", epoch + 1, step, loss);
                }
            }
        }
        Ok(())
    }
    fn compute_class_centroids(&self, embed: &Sequential) -> CategoryResult<HashMap<u8, Tensor>> {
        let mut sums: HashMap<u8, Tensor> = HashMap::new();
        let mut counts: HashMap<u8, usize> = HashMap::new();

        for batch in self.dataset.batch_view_iter(100, self.dataset.labels.len()) {
            let images = batch.images_tensor(&self.device)?;
            let (emb, _) = embed.forward_with_tape(images)?; // (batch, dim)

            for (row_idx, &label) in batch.labels.iter().enumerate() {
                let row = emb.narrow(0, row_idx, 1)?; // (1, dim)
                sums.entry(label)
                    .and_modify(|acc| {
                        *acc = (&*acc + &row).unwrap();
                    })
                    .or_insert(row);
                *counts.entry(label).or_insert(0) += 1;
            }
        }

        let mut centroids = HashMap::with_capacity(sums.len());
        for (label, sum) in sums {
            let n = counts[&label] as f64;
            centroids.insert(label, sum.affine(1.0 / n, 0.0)?);
        }
        Ok(centroids)
    }

    pub fn test(&self, embed: &Sequential, testset: DataSet) -> CategoryResult<()> {
        let centroids = self.compute_class_centroids(embed)?;

        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for (batch_idx, batch) in testset.batch_view_iter(100, 10_000).enumerate() {
            let images = batch.images_tensor(&self.device)?;
            let (emb, _) = embed.forward_with_tape(images)?; // (batch, dim)
            let batch_size = emb.dim(0)?;

            let mut correct_in_batch = 0usize;

            for row_idx in 0..batch_size {
                let sample = emb.narrow(0, row_idx, 1)?; // (1, dim)

                let mut best_label: Option<u8> = None;
                let mut best_dist = f64::INFINITY;

                for (&label, centroid) in centroids.iter() {
                    let diff = (&sample - centroid)?;
                    let dist = diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                    if dist < best_dist {
                        best_dist = dist;
                        best_label = Some(label);
                    }
                }

                if best_label == Some(batch.labels[row_idx]) {
                    correct_in_batch += 1;
                }
            }

            total_correct += correct_in_batch;
            total_samples += batch_size;

            println!("batch:              {}", batch_idx);
            println!("correct (batch):    {}", correct_in_batch);
        }

        let accuracy = total_correct as f64 / total_samples as f64 * 100.0;
        println!("Accuracy = {:.2}%", accuracy);
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
