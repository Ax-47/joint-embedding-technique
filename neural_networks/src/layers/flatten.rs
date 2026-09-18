use std::rc::Rc;

use candle_core::Tensor;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};

pub struct FlattenLayer;

impl Morphism for FlattenLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "flatten layer"
    }

    fn apply(&self, input: Tensor) -> CategoryResult<Tensor> {
        let n = input.dims()[0];
        Ok(input.reshape((n, ()))?.contiguous()?)
    }
}

pub struct DerivativeFlatten {
    pub input_shape: Vec<usize>,
}

impl Morphism for DerivativeFlatten {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "flatten derivative"
    }

    fn apply(&self, grad_output: Tensor) -> CategoryResult<Tensor> {
        Ok(grad_output.reshape(self.input_shape.as_slice())?)
    }
}

impl DerivativeMorphism for FlattenLayer {
    type Input = Tensor;
    type Curry = Tensor;
    type Output = Tensor;

    fn derivative(&self, input: Self::Curry) -> Rc<dyn Morphism<Input = Tensor, Output = Tensor>> {
        Rc::new(DerivativeFlatten {
            input_shape: input.dims().to_vec(),
        })
    }
}
