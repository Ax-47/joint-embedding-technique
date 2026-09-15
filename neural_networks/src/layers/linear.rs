use std::rc::Rc;

use utils::currying_morphism::DerivatibleCurryingMorphism;
use utils::derivative::DerivatibleMorphism;
use utils::derivative::DerivativeMorphism;
use utils::morphism::Morphism;

use ndarray::{Array1, Array2, Axis};

use crate::layers::layer::Layer;
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    pub weight_matrix: Array2<f64>,
    pub bias_matrix: Array1<f64>,
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
    type Input = Array2<f64>;
    type Output = Array2<f64>;

    type DerivativeInput = (Array2<f64>, Array2<f64>);
    type DerivativeOutput = (LinearLayerParams, Array2<f64>);
    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<
        dyn DerivatibleMorphism<
                Self::Input,
                Self::Output,
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
    type Input = Array2<f64>;
    type Output = Array2<f64>;
    fn name(&self) -> &'static str {
        "linear layer (applied)"
    }
    fn apply(&self, input: Self::Input) -> utils::errors::CategoryResult<Self::Output> {
        Ok(input.dot(&self.params.weight_matrix) + &self.params.bias_matrix)
    }
}

pub struct DerivativeLinearLayer {
    pub params: LinearLayerParams,
}
impl Morphism for DerivativeLinearLayer {
    type Input = (Array2<f64>, Array2<f64>);
    type Output = (LinearLayerParams, Array2<f64>);
    fn name(&self) -> &'static str {
        "derivative linear layer"
    }
    fn apply(
        &self,
        (input, grad_output): Self::Input,
    ) -> utils::errors::CategoryResult<Self::Output> {
        let dw = input.t().dot(&grad_output);
        let db = grad_output.sum_axis(Axis(0));
        let da = grad_output.dot(&self.params.weight_matrix.t());

        Ok((
            LinearLayerParams {
                weight_matrix: dw,
                bias_matrix: db,
            },
            da,
        ))
    }
}

impl DerivativeMorphism for AppliedLinearLayer {
    type Input = (Array2<f64>, Array2<f64>);
    type Output = (LinearLayerParams, Array2<f64>);
    fn derivative(&self) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeLinearLayer {
            params: self.params.clone(),
        })
    }
}
