use std::rc::Rc;

use candle_core::Tensor;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};

pub struct CosineSimilarity {
    eps: f32,
}
impl Default for CosineSimilarity {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

impl Morphism for CosineSimilarity {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "CosineSimilarity"
    }

    fn apply(&self, (left, right): Self::Input) -> CategoryResult<Self::Output> {
        let eps = Tensor::new(self.eps, left.device())?;

        let left_norm = left.sqr()?.sum(1)?.sqrt()?.broadcast_add(&eps)?;
        let right_norm = right.sqr()?.sum(1)?.sqrt()?.broadcast_add(&eps)?;

        let dot = left.broadcast_mul(&right)?.sum(1)?;
        let cos = dot.broadcast_div(&left_norm.broadcast_mul(&right_norm)?)?;

        Ok(cos)
    }
}
pub struct DerivativeCosineSimilarity {
    eps: f32,
}

impl DerivativeMorphism for CosineSimilarity {
    type Input = (Tensor, Tensor);
    type Output = (Tensor, Tensor);
    type Curry = ();

    fn derivative(
        &self,
        _input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeCosineSimilarity { eps: self.eps })
    }
}

impl Morphism for DerivativeCosineSimilarity {
    type Input = (Tensor, Tensor);
    type Output = (Tensor, Tensor);

    fn name(&self) -> &'static str {
        "DerivativeCosineSimilarity"
    }

    fn apply(&self, (left, right): Self::Input) -> CategoryResult<Self::Output> {
        let eps = Tensor::new(self.eps, left.device())?;

        let left_norm = left.sqr()?.sum(1)?.sqrt()?.broadcast_add(&eps)?;
        let right_norm = right.sqr()?.sum(1)?.sqrt()?.broadcast_add(&eps)?;

        let left_hat = left.broadcast_div(&left_norm.unsqueeze(1)?)?;
        let right_hat = right.broadcast_div(&right_norm.unsqueeze(1)?)?;

        let cos = left_hat.broadcast_mul(&right_hat)?.sum(1)?;
        let cos_exp = cos.unsqueeze(1)?;

        let grad_left = right_hat
            .broadcast_sub(&left_hat.broadcast_mul(&cos_exp)?)?
            .broadcast_div(&left_norm.unsqueeze(1)?)?;

        let grad_right = left_hat
            .broadcast_sub(&right_hat.broadcast_mul(&cos_exp)?)?
            .broadcast_div(&right_norm.unsqueeze(1)?)?;

        Ok((grad_left, grad_right))
    }
}
