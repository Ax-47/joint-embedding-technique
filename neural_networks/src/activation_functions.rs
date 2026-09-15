use candle_core::Tensor;
use core::f32;
use std::rc::Rc;
use utils::{
    derivative::{DerivatibleMorphism, DerivativeMorphism},
    errors::CategoryResult,
    morphism::Morphism,
};

pub struct Relu;
impl DerivativeMorphism for Relu {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn derivative(&self) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeRelu)
    }
}
pub struct DerivativeRelu;

impl Morphism for Relu {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "relu"
    }

    fn apply(&self, input: Tensor) -> CategoryResult<Self::Output> {
        Ok(input.relu()?)
    }
}

impl Morphism for DerivativeRelu {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "derivative relu"
    }

    fn apply(&self, (input, grad_output): (Tensor, Tensor)) -> CategoryResult<Self::Output> {
        let mask = input.gt(0.0)?;
        let mask = mask.to_dtype(grad_output.dtype())?;

        Ok(grad_output.broadcast_mul(&mask)?)
    }
}
pub struct LeakyRelu {
    alpha: f32,
}

pub struct Sigmoid;

impl Morphism for Sigmoid {
    type Input = Tensor;
    type Output = Tensor;
    fn name(&self) -> &'static str {
        "sigmoid"
    }
    fn apply(&self, x: Tensor) -> utils::errors::CategoryResult<Tensor> {
        Ok(x.sign()?)
    }
}

pub struct SigmoidDerivative;

impl Morphism for SigmoidDerivative {
    type Input = Tensor;
    type Output = Tensor;
    fn name(&self) -> &'static str {
        "sigmoid derivative"
    }
    fn apply(&self, x: Tensor) -> utils::errors::CategoryResult<Tensor> {
        let y = x.sign()?;
        Ok((&y * (1.0 - &y)?)?)
    }
}

impl DerivativeMorphism for Sigmoid {
    type Input = Tensor;
    type Output = Tensor;
    fn derivative(&self) -> Rc<dyn Morphism<Input = Tensor, Output = Tensor>> {
        Rc::new(SigmoidDerivative)
    }
}
