use std::rc::Rc;

use candle_core::Tensor;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};

pub struct MSELoss;

impl Morphism for MSELoss {
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

pub struct DerivativeMSELoss;
impl DerivativeMorphism for MSELoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;
    type Curry = ();
    fn derivative(
        &self,
        _input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeMSELoss)
    }
}
impl Morphism for DerivativeMSELoss {
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
