use candle_core::{Result, Tensor};
use smallvec::SmallVec;

/// layer ส่วนใหญ่มี weight + bias → 2 slot อยู่บน stack ได้เลย
pub type TensorList = SmallVec<[Tensor; 4]>;

#[derive(Clone)]
pub struct ParamSet {
    pub tensors: TensorList,
}

impl ParamSet {
    pub fn new<const N: usize>(tensors: [Tensor; N]) -> Self {
        Self {
            tensors: tensors.into_iter().collect(),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> &Tensor {
        &self.tensors[index]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl From<TensorList> for ParamSet {
    #[inline]
    fn from(tensors: TensorList) -> Self {
        Self { tensors }
    }
}
impl ParamSet {
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a.add(b))
    }

    pub fn sgd(&self, grads: &Self, learning_rate: f64) -> Result<Self> {
        self.zip_with(grads, |param, grad| {
            let step = grad.affine(-learning_rate, 0.0)?;
            param.add(&step)
        })
    }

    fn zip_with<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    {
        if self.len() != other.len() {
            return Err(candle_core::Error::Msg(format!(
                "param count mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        let mut out = TensorList::with_capacity(self.len());
        for (a, b) in self.tensors.iter().zip(other.tensors.iter()) {
            out.push(f(a, b)?);
        }

        Ok(Self { tensors: out })
    }
}
