use std::cell::RefCell;
use std::rc::Rc;

use utils::derivative::DerivativeMorphism;
use utils::morphism::{CurryingMorphism, Morphism};

use ndarray::{Array1, Array2, Axis};
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    pub weight_matrix: Array2<f64>,
    pub bias_matrix: Array1<f64>,
}
pub type SharedParams = Rc<RefCell<LinearLayerParams>>;

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

pub struct AppliedLinearLayer {
    params: LinearLayerParams,
}

impl Morphism for AppliedLinearLayer {
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    fn name(&self) -> &'static str {
        "linear layer (applied)"
    }
    fn apply(&self, input: Self::Input) -> utils::errors::CategoryResult<Self::Output> {
        Ok(self.params.weight_matrix.t().dot(&input) + &self.params.bias_matrix)
    }
}

impl CurryingMorphism for LinearLayer {
    type Params = LinearLayerParams;
    type Input = Array1<f64>;
    type Output = Array1<f64>;
    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(AppliedLinearLayer {
            params: params.clone(),
        })
    }
}

pub struct DerivativeLinearLayer {
    pub params: LinearLayerParams,
}
impl Morphism for DerivativeLinearLayer {
    type Input = (Array1<f64>, Array1<f64>);
    type Output = (LinearLayerParams, Array1<f64>);
    fn name(&self) -> &'static str {
        "derivative linear layer"
    }
    fn apply(
        &self,
        (input, grad_output): Self::Input,
    ) -> utils::errors::CategoryResult<Self::Output> {
        let dw = input
            .clone()
            .insert_axis(Axis(1))
            .dot(&grad_output.clone().insert_axis(Axis(0)));
        let db = grad_output.clone();
        let da = self.params.weight_matrix.dot(&grad_output);

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
    type Input = (Array1<f64>, Array1<f64>);
    type Output = (LinearLayerParams, Array1<f64>);
    fn derivative(&self) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeLinearLayer {
            params: self.params.clone(),
        })
    }
}
