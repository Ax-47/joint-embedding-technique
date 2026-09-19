use candle_core::{Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::layers::param_set::ParamSet;
use neural_networks::{container::sequential::Sequential, layers::linear::LinearLayerParams};
use std::collections::HashMap;
use std::io::Write;

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

pub struct BarlowTrainStep {
    dataset: DataSet,
    learning_rate: f64,
    batch_size: usize,
    cut_shape: (usize, usize),
    device: Device,
}

impl BarlowTrainStep {
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
        let lambda = 5e-3; // ค่าต้นฉบับ Barlow Twins ใช้ 5e-3
        let batch_size_f = self.batch_size as f64;

        for epoch in 0..10 {
            for step in 0..steps_per_epoch {
                // pair.left / pair.right คือ augmentation คู่จากรูปเดียวกัน
                let pair =
                    self.dataset
                        .sample_pairs(self.batch_size, self.cut_shape, &self.device)?;

                // Forward ทั้งสองฝั่ง (weights 共享)
                let (z_a, tape_a) = embed.forward_with_tape(pair.left)?;
                let (z_b, tape_b) = embed.forward_with_tape(pair.right)?;

                // z_a, z_b shape: [batch_size, num_features]
                // 1. Center ตาม batch dimension (dim=0)
                let mean_a = z_a.mean_keepdim(0)?; // [1, D]
                let mean_b = z_b.mean_keepdim(0)?; // [1, D]

                let za_c = z_a.broadcast_sub(&mean_a)?; // [N, D]
                let zb_c = z_b.broadcast_sub(&mean_b)?; // [N, D]

                // 2. Cross-covariance matrix C = (Z_a^T @ Z_b) / N
                // shape: [D, D]
                let c = za_c.t()?.matmul(&zb_c)?;
                let c = c.affine(1.0 / batch_size_f, 0.0)?;

                // 3. Barlow Twins Loss
                let num_features = z_a.dim(1)?;
                let identity = Tensor::eye(num_features, z_a.dtype(), &self.device)?;

                // Invariance: (diag(C) - 1)^2
                let c_diag = c.broadcast_mul(&identity)?; // zero out off-diagonal
                let invariance = c_diag.sub(&identity)?.sqr()?.sum_all()?;

                // Redundancy: sum of off-diagonal squared
                let c_sq = c.sqr()?;
                let total_sq = c_sq.sum_all()?;
                let diag_sq = c_diag.sqr()?.sum_all()?;
                let redundancy = total_sq.sub(&diag_sq)?;

                let loss = invariance.add(&redundancy.affine(lambda, 0.0)?)?;

                // 4. Manual backward: คำนวณ gradient ของ loss w.r.t z_a, z_b
                // สร้าง matrix M ที่ dL/dC = M
                // M_ii = 2(C_ii - 1), M_ij = 2*lambda*C_ij (i≠j)
                let off_diag_mask = {
                    let ones = Tensor::ones_like(&c)?;
                    ones.sub(&identity)?
                };

                let m_diag = c_diag.sub(&identity)?.affine(2.0, 0.0)?;
                let m_offdiag = c.broadcast_mul(&off_diag_mask)?.affine(2.0 * lambda, 0.0)?;
                let m = m_diag.add(&m_offdiag)?; // [D, D]

                // dL/dza_c = (1/N) * zb_c @ M^T
                // dL/dzb_c = (1/N) * za_c @ M
                let scale = 1.0 / batch_size_f;
                let dza_c = zb_c.matmul(&m.t()?)?.affine(scale, 0.0)?;
                let dzb_c = za_c.matmul(&m)?.affine(scale, 0.0)?;

                // 5. Backprop through centering: za_c = za - mean(za)
                // dL/dza = dL/dza_c - mean(dL/dza_c)
                let mean_dza = dza_c.mean_keepdim(0)?;
                let delta_a = dza_c.broadcast_sub(&mean_dza)?;

                let mean_dzb = dzb_c.mean_keepdim(0)?;
                let delta_b = dzb_c.broadcast_sub(&mean_dzb)?;

                // 6. Backprop through encoder
                let grads_a = embed.backward_with_tape(tape_a, delta_a)?;
                let grads_b = embed.backward_with_tape(tape_b, delta_b)?;
                let grads = add_all(&grads_a, &grads_b)?;

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
            let emb = embed.forward(images)?; // (batch, dim)

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

    pub fn test(&self, embed: &mut Sequential, testset: DataSet) -> CategoryResult<()> {
        let centroids = self.compute_class_centroids(embed)?;

        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for (batch_idx, batch) in testset.batch_view_iter(100, 10_000).enumerate() {
            let images = batch.images_tensor(&self.device)?;
            let emb = embed.forward(images)?; // (batch, dim)
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
