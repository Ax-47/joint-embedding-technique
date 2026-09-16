use candle_core::{DType, Device, Tensor};
use utils::{errors::CategoryResult, morphism::Morphism};

pub struct Loss;

impl Morphism for Loss {
    type Input = (Tensor, Tensor);
    type Output = f32;

    fn name(&self) -> &'static str {
        "Loss"
    }

    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        let diff = (&al - &y)?;
        let loss = diff.sqr()?.sum_all()?.to_scalar::<f32>()?;
        Ok(loss / al.elem_count() as f32)
    }
}

pub struct DerivativeLoss;
impl Morphism for DerivativeLoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "DerivativeLoss"
    }

    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        let n = al.elem_count() as f64;
        Ok((&al - &y)?.affine(2.0 / n, 0.0)?)
    }
}
