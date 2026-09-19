use candle_core::Tensor;
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
