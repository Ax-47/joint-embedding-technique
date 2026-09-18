use crate::read_byte::DataSet;
use candle_core::{Device, Tensor}; //pair_read_byte.rs
use image::{GrayImage, ImageBuffer, Luma, RgbImage};
use rand::RngExt;
pub struct PairBatch {
    pub left: Tensor,
    pub right: Tensor,
    pub labels: Tensor,
}

impl DataSet {
    pub fn sample_pairs(
        &self,
        batch_size: usize,
        cut_shape: (usize, usize),
        device: &Device,
    ) -> candle_core::Result<PairBatch> {
        let mut rng = rand::rng();

        let classes: Vec<u8> = self.by_label.keys().copied().collect();

        let mut left_idx = Vec::with_capacity(batch_size);
        let mut right_idx = Vec::with_capacity(batch_size);
        let mut pair_labels = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let anchor_class = classes[rng.random_range(0..classes.len())];
            let anchor_pool = &self.by_label[&anchor_class];
            let a = anchor_pool[rng.random_range(0..anchor_pool.len())];

            let same = i % 2 == 0;

            let b = if same {
                loop {
                    let cand = anchor_pool[rng.random_range(0..anchor_pool.len())];
                    if cand != a || anchor_pool.len() == 1 {
                        break cand;
                    }
                }
            } else {
                loop {
                    let other = classes[rng.random_range(0..classes.len())];
                    if other != anchor_class {
                        let pool = &self.by_label[&other];
                        break pool[rng.random_range(0..pool.len())];
                    }
                }
            };

            left_idx.push(a);
            right_idx.push(b);
            pair_labels.push(if same { 1.0f32 } else { 0.0f32 });
        }

        let left = self.cutout_images_by_index(&left_idx, cut_shape, device)?;
        let right = self.cutout_images_by_index(&right_idx, cut_shape, device)?;
        let labels = Tensor::from_vec(pair_labels, batch_size, device)?;

        Ok(PairBatch {
            left,
            right,
            labels,
        })
    }

    fn cutout_images_by_index(
        &self,
        indices: &[usize],
        cut_shape: (usize, usize),
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let (img_h, img_w) = self.image_shape;
        let (cut_h, cut_w) = cut_shape;

        let max_y = img_h.checked_sub(cut_h).expect("cut_h ใหญ่กว่าความสูงภาพจริง");
        let max_x = img_w
            .checked_sub(cut_w)
            .expect("cut_w ใหญ่กว่าความกว้างภาพจริง");

        let mut rng = rand::rng();
        let mut buf = Vec::with_capacity(indices.len() * img_h * img_w);

        for &idx in indices {
            let row = self.images.row(idx);
            let row_slice = row.as_slice().expect("row ต้อง contiguous");

            let off_y = if max_y == 0 {
                0
            } else {
                rng.random_range(0..=max_y)
            };
            let off_x = if max_x == 0 {
                0
            } else {
                rng.random_range(0..=max_x)
            };

            let mut img: Vec<f32> = row_slice.iter().map(|p| *p as f32 / 255.0).collect();

            for y in off_y..off_y + cut_h {
                let start = y * img_w + off_x;
                for px in &mut img[start..start + cut_w] {
                    *px = 0.0;
                }
            }

            buf.extend(img);
        }

        Tensor::from_vec(buf, (indices.len(), img_h * img_w), device)
    }
    pub fn export_pair_preview(
        &self,
        n_pairs: usize,
        cut_shape: (usize, usize),
        out_path: &str,
    ) -> candle_core::Result<()> {
        let device = candle_core::Device::Cpu;
        let pair = self.sample_pairs(n_pairs, cut_shape, &device)?;

        let (img_h, img_w) = self.image_shape;
        let left_data: Vec<Vec<f32>> = pair.left.to_vec2()?;
        let right_data: Vec<Vec<f32>> = pair.right.to_vec2()?;
        let labels: Vec<f32> = pair.labels.to_vec1()?;

        let pad = 4; // ช่องว่างระหว่างภาพ + ที่เว้นให้ border
        let cell_w = img_w * 2 + pad * 3; // left + right + ขอบ 3 ด้าน
        let cell_h = img_h + pad * 2;

        let canvas_w = cell_w as u32;
        let canvas_h = (cell_h * n_pairs) as u32;

        let mut canvas: RgbImage =
            ImageBuffer::from_pixel(canvas_w, canvas_h, image::Rgb([30, 28, 22]));

        for i in 0..n_pairs {
            let same = labels[i] > 0.5;
            let border_color = if same {
                image::Rgb([58, 90, 64]) // เขียว = same
            } else {
                image::Rgb([182, 84, 60]) // แดง = diff
            };

            let row_y0 = i * cell_h;

            // วาด border รอบ cell ทั้งแถว
            for x in 0..cell_w {
                for t in 0..pad {
                    canvas.put_pixel(x as u32, (row_y0 + t) as u32, border_color);
                    canvas.put_pixel(x as u32, (row_y0 + cell_h - 1 - t) as u32, border_color);
                }
            }

            // วาดภาพ left
            draw_gray_f32(&mut canvas, &left_data[i], img_w, img_h, pad, row_y0 + pad);
            // วาดภาพ right (offset ไปทางขวา)
            draw_gray_f32(
                &mut canvas,
                &right_data[i],
                img_w,
                img_h,
                pad * 2 + img_w,
                row_y0 + pad,
            );
        }

        canvas
            .save(out_path)
            .map_err(|e| candle_core::Error::wrap(e))?;
        Ok(())
    }
}
fn draw_gray_f32(
    canvas: &mut RgbImage,
    data: &[f32],
    w: usize,
    h: usize,
    off_x: usize,
    off_y: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let v = (data[y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
            canvas.put_pixel(
                (off_x + x) as u32,
                (off_y + y) as u32,
                image::Rgb([v, v, v]),
            );
        }
    }
}
