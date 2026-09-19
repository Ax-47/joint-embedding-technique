use candle_core::Tensor;
use std::rc::Rc;
use utils::{derivative::DerivativeMorphism, morphism::Morphism};
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
