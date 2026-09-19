use candle_core::{Device, Tensor}; //read_byte.rs
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray::{Axis, s};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Result};
#[derive(Clone)]
pub struct DataSet {
    pub labels: Array1<u8>,
    pub images: Array2<u8>,
    pub image_shape: (usize, usize), // (rows, cols)
    pub(crate) by_label: HashMap<u8, Vec<usize>>,
}

pub struct BatchView<'a> {
    pub images: ArrayView2<'a, u8>, // (batch, 784)
    pub labels: ArrayView1<'a, u8>, // (batch)
}

impl<'a> BatchView<'a> {
    pub fn images_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let (batch, dim) = self.images.dim();
        let mut buf = Vec::with_capacity(batch * dim);

        for row in self.images.rows() {
            buf.extend(row.iter().map(|p| *p as f32 / 255.0));
        }

        Tensor::from_vec(buf, (batch, dim), device)
    }

    pub fn label_one_hot_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let batch = self.labels.len();
        let mut buf = vec![0.0f32; batch * 10];

        for (i, &label) in self.labels.iter().enumerate() {
            buf[i * 10 + label as usize] = 1.0;
        }

        Tensor::from_vec(buf, (batch, 10), device)
    }

    pub fn labels_u32_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let buf: Vec<u32> = self.labels.iter().map(|&l| l as u32).collect();
        Tensor::from_vec(buf, self.labels.len(), device)
    }
}

impl DataSet {
    pub fn new(image_path: &str, label_path: &str) -> Result<Self> {
        let (images, image_shape) = Self::load_images_ndarray(image_path)?;
        let labels = Self::load_labels(label_path)?;

        assert_eq!(images.nrows(), labels.len());

        let by_label = Self::build_by_label(&labels);

        Ok(Self {
            images,
            labels,
            image_shape,
            by_label,
        })
    }

    pub fn shuffle(&mut self) {
        self.shuffle_with_rng(&mut rand::rng());
    }

    pub fn shuffle_with_seed(&mut self, seed: u64) {
        self.shuffle_with_rng(&mut StdRng::seed_from_u64(seed));
    }

    fn shuffle_with_rng(&mut self, rng: &mut impl rand::Rng) {
        let n = self.labels.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(rng);

        self.images = self.images.select(Axis(0), &idx);
        self.labels = Array1::from_vec(idx.iter().map(|&i| self.labels[i]).collect());
        self.by_label = Self::build_by_label(&self.labels);
    }
    fn build_by_label(labels: &Array1<u8>) -> HashMap<u8, Vec<usize>> {
        let mut by_label: HashMap<u8, Vec<usize>> = HashMap::new();
        for (i, &l) in labels.iter().enumerate() {
            by_label.entry(l).or_default().push(i);
        }
        by_label
    }

    fn read_u32_be(buf: &[u8]) -> u32 {
        u32::from_be_bytes(buf.try_into().unwrap())
    }

    fn get_buf_from_file(path: &str) -> Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn load_labels(path: &str) -> Result<Array1<u8>> {
        let buf = Self::get_buf_from_file(path)?;

        let magic = Self::read_u32_be(&buf[0..4]);
        assert_eq!(magic, 2049);

        let num = Self::read_u32_be(&buf[4..8]) as usize;
        let arr = Array1::from_vec(buf[8..8 + num].to_vec());
        Ok(arr)
    }

    fn load_images_ndarray(path: &str) -> Result<(Array2<u8>, (usize, usize))> {
        let buf = Self::get_buf_from_file(path)?;

        let magic = Self::read_u32_be(&buf[0..4]);
        assert_eq!(magic, 2051);

        let num = Self::read_u32_be(&buf[4..8]) as usize;
        let rows = Self::read_u32_be(&buf[8..12]) as usize;
        let cols = Self::read_u32_be(&buf[12..16]) as usize;

        let image_size = rows * cols;
        let offset = 16;

        let data = &buf[offset..offset + num * image_size];

        let arr = Array2::from_shape_vec((num, image_size), data.to_vec()).unwrap();
        Ok((arr, (rows, cols)))
    }

    pub fn batch_view_iter(
        &self,
        batch_size: usize,
        limit: usize,
    ) -> impl Iterator<Item = BatchView<'_>> {
        let n = self.labels.len().min(limit);

        (0..n).step_by(batch_size).map(move |start| {
            let end = (start + batch_size).min(n);

            BatchView {
                labels: self.labels.slice(s![start..end]),
                images: self.images.slice(s![start..end, ..]),
            }
        })
    }
}
