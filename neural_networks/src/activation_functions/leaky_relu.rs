use candle_core::Tensor;
use std::rc::Rc;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu {
    pub alpha: f32,
}

impl LeakyRelu {
    pub fn new(alpha: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&alpha),
            "LeakyReLU alpha should be in [0, 1)"
        );

        Self { alpha }
    }
}

impl DerivativeMorphism for LeakyRelu {
    type Input = Tensor;
    type Output = Tensor;
    type Curry = Tensor;
    fn derivative(
        &self,
        input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeLeakyRelu {
            input,
            alpha: self.alpha,
        })
    }
}

impl Morphism for LeakyRelu {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "relu"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        let positive_mask = input.gt(0.0)?.to_dtype(input.dtype())?;
        let negative_mask = Tensor::ones_like(&positive_mask)?.sub(&positive_mask)?;
        let positive_part = input.broadcast_mul(&positive_mask)?;
        let negative_part = input
            .broadcast_mul(&negative_mask)?
            .affine(self.alpha as f64, 0.0)?;

        Ok(positive_part.add(&negative_part)?)
    }
}

pub struct DerivativeLeakyRelu {
    pub input: Tensor,
    pub alpha: f32,
}

impl Morphism for DerivativeLeakyRelu {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "derivative leaky relu"
    }

    fn apply(&self, grad: Self::Input) -> CategoryResult<Self::Output> {
        let positive_mask = self.input.gt(0.0)?.to_dtype(grad.dtype())?;
        let negative_mask = Tensor::ones_like(&positive_mask)?.sub(&positive_mask)?;
        let negative_slope = negative_mask.affine(self.alpha as f64, 0.0)?;
        let slope = positive_mask.add(&negative_slope)?;
        Ok(grad.broadcast_mul(&slope)?)
    }
}
