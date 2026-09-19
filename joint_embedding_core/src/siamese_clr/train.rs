use candle_core::{DType, Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::layers::param_set::ParamSet;
use neural_networks::{container::sequential::Sequential, layers::linear::LinearLayerParams};
use std::collections::HashMap;
use std::io::Write;

use utils::errors::{CategoryError, CategoryResult};
const EPS_LOG: f64 = 1e-8;
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

pub struct SiameseCLRTrainStep {
    dataset: DataSet,
    learning_rate: f64,
    batch_size: usize,
    cut_shape: (usize, usize),
    temperature: f64,
    device: Device,
}

impl SiameseCLRTrainStep {
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
            temperature: 0.5,
            device,
        }
    }

    pub fn train(&self, embed: &mut Sequential) -> CategoryResult<()> {
        let steps_per_epoch = self.dataset.labels.len() / self.batch_size;
        let b = self.batch_size;
        let two_b = 2 * b;

        for epoch in 0..10 {
            for step in 0..steps_per_epoch {
                let views = self.dataset.sample_views(b, self.cut_shape, &self.device)?;

                let left_emb = embed.forward(views.left)?;
                let right_emb = embed.forward(views.right)?;

                // รวมเป็น (2B, dim): แถว 0..B = left view, แถว B..2B = right view
                let all_emb = Tensor::cat(&[&left_emb, &right_emb], 0)?;

                // L2 normalize แต่ละแถว (cosine similarity ต้องการเวกเตอร์หนึ่งหน่วย)
                let norm = all_emb.sqr()?.sum_keepdim(1)?.sqrt()?; // (2B, 1)
                let z = all_emb.broadcast_div(&norm)?; // (2B, dim)

                // similarity matrix (2B, 2B) = z @ z^T / temperature
                let sim = z.matmul(&z.t()?)?.affine(1.0 / self.temperature, 0.0)?;

                // มาสก์ diagonal เป็น -inf กัน anchor เทียบกับตัวเอง
                let neg_inf_diag =
                    Tensor::eye(two_b, DType::F32, &self.device)?.affine(-1e9, 0.0)?;
                let sim_masked = (&sim + &neg_inf_diag)?;

                // softmax ต่อแถว (numerically stable)
                let sim_max = sim_masked.max_keepdim(1)?;
                let exp_sim = sim_masked.broadcast_sub(&sim_max)?.exp()?;
                let sum_exp = exp_sim.sum_keepdim(1)?;
                let p = exp_sim.broadcast_div(&sum_exp)?; // (2B, 2B)

                // positive index ของแถว i คือ (i+B) mod 2B (คู่ view เดียวกัน)
                let mut onehot_buf = vec![0f32; two_b * two_b];
                for i in 0..two_b {
                    onehot_buf[i * two_b + (i + b) % two_b] = 1.0;
                }
                let onehot = Tensor::from_vec(onehot_buf, (two_b, two_b), &self.device)?;

                // loss สำหรับ log
                let log_p_pos = (&p * &onehot)?.sum(1)?.affine(1.0, EPS_LOG)?.log()?;
                let loss = log_p_pos.neg()?.mean_all()?;

                // --- backward แบบ derive มือ (standard softmax-CE gradient) ---
                // dL/dS = (P - onehot) / (2B)
                let g = (&p - &onehot)?.affine(1.0 / two_b as f64, 0.0)?;
                let g_sym = (&g + &g.t()?)?;

                // dL/dz = (1/tau) * (G + G^T) @ Z
                let dz = g_sym.matmul(&z)?.affine(1.0 / self.temperature, 0.0)?;

                // backprop ผ่าน L2 normalize: de = (dz - z*(z·dz)) / ||e||
                let dot = (&z * &dz)?.sum_keepdim(1)?;
                let correction = z.broadcast_mul(&dot)?;
                let de = (&dz - &correction)?.broadcast_div(&norm)?; // (2B, dim)

                let delta_left = de.narrow(0, 0, b)?;
                let delta_right = de.narrow(0, b, b)?;

                let grads_left = embed.backward(delta_left)?;
                let grads_right = embed.backward(delta_right)?;
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
