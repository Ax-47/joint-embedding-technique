use candle_core::Tensor;
use core::f32;
use std::rc::Rc;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};

pub struct Relu;
impl DerivativeMorphism for Relu {
    type Input = Tensor;
    type Output = Tensor;
    type Curry = Tensor;
    fn derivative(
        &self,
        output: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeRelu { output })
    }
}
pub struct DerivativeRelu {
    output: Tensor,
}

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
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "derivative relu"
    }

    fn apply(&self, grad: Tensor) -> CategoryResult<Self::Output> {
        let mask = &self.output.gt(0.0)?;
        let mask = mask.to_dtype(grad.dtype())?;
        Ok(grad.broadcast_mul(&mask)?)
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

pub struct SigmoidDerivative {
    output: Tensor,
}

impl Morphism for SigmoidDerivative {
    type Input = ();
    type Output = Tensor;
    fn name(&self) -> &'static str {
        "sigmoid derivative"
    }
    fn apply(&self, _x: ()) -> utils::errors::CategoryResult<Tensor> {
        let y = self.output.clone();
        Ok((&y * (1.0 - &y)?)?)
    }
}

impl DerivativeMorphism for Sigmoid {
    type Input = ();
    type Curry = Tensor;
    type Output = Tensor;
    fn derivative(
        &self,
        output: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(SigmoidDerivative { output })
    }
}
