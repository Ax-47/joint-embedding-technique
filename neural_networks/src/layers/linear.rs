use candle_core::Tensor;
use std::rc::Rc;
use utils::currying_morphism::DerivatibleCurryingMorphism;
use utils::derivative::DerivatibleMorphism;
use utils::derivative::DerivativeMorphism;
use utils::morphism::Morphism;

use crate::layers::layer::Layer;
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    pub weight_matrix: Tensor,
    pub bias_matrix: Tensor,
}

pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,
}
impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
        }
    }
}
impl Layer for LinearLayer {
    fn features(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }
}
pub struct AppliedLinearLayer {
    params: LinearLayerParams,
}

impl DerivatibleCurryingMorphism for LinearLayer {
    type Params = LinearLayerParams;
    type Input = Tensor;
    type Output = Tensor;
    type Curry = Tensor;
    type DerivativeInput = Tensor;
    type DerivativeOutput = (LinearLayerParams, Tensor);
    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<
        dyn DerivatibleMorphism<
                Self::Input,
                Self::Output,
                Self::Curry,
                Self::DerivativeInput,
                Self::DerivativeOutput,
            >,
    > {
        Rc::new(AppliedLinearLayer {
            params: params.clone(),
        })
    }
}

impl Morphism for AppliedLinearLayer {
    type Input = Tensor;
    type Output = Tensor;
    fn name(&self) -> &'static str {
        "linear layer (applied)"
    }
    fn apply(&self, input: Self::Input) -> utils::errors::CategoryResult<Self::Output> {
        let y = input.matmul(&self.params.weight_matrix)?;
        Ok(y.broadcast_add(&self.params.bias_matrix)?)
    }
}

pub struct DerivativeLinearLayer {
    pub params: LinearLayerParams,
    pub output: Tensor,
}
impl Morphism for DerivativeLinearLayer {
    type Input = Tensor;
    type Output = (LinearLayerParams, Tensor);
    fn name(&self) -> &'static str {
        "derivative linear layer "
    }
    fn apply(&self, grad_output: Self::Input) -> utils::errors::CategoryResult<Self::Output> {
        let dw = &self.output.t()?.matmul(&grad_output)?;
        let db = grad_output.sum(0)?;
        let da = grad_output.matmul(&self.params.weight_matrix.t()?)?;

        Ok((
            LinearLayerParams {
                weight_matrix: dw.clone(),
                bias_matrix: db,
            },
            da,
        ))
    }
}

impl DerivativeMorphism for AppliedLinearLayer {
    type Input = Tensor;
    type Curry = Tensor;
    type Output = (LinearLayerParams, Tensor);
    fn derivative(
        &self,
        output: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeLinearLayer {
            params: self.params.clone(),
            output,
        })
    }
}
