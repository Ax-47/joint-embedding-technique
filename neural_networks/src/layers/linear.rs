use std::cell::RefCell;
use std::rc::Rc;

use utils::morphism::CurryingMorphism;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    pub weight_matrix: Array2<f64>,
    pub bias_matrix: Array1<f64>,
}
type LinearLayerForwardFn = fn(&SharedParams, Array1<f64>) -> Array1<f64>;
type LinearLayerBackwardFn =
    fn(&SharedParams, (Array1<f64>, Array1<f64>)) -> (LinearLayerParams, Array1<f64>);
pub type LinearLayerForward =
    CurryingMorphism<LinearLayerForwardFn, SharedParams, Array1<f64>, Array1<f64>>;
pub type LinearLayerBackward = CurryingMorphism<
    LinearLayerBackwardFn,
    SharedParams,
    (Array1<f64>, Array1<f64>),
    (LinearLayerParams, Array1<f64>),
>;
pub type SharedParams = Rc<RefCell<LinearLayerParams>>;
fn forward(params: &SharedParams, input: Array1<f64>) -> Array1<f64> {
    params.borrow().weight_matrix.t().dot(&input) + &params.borrow().bias_matrix
}

fn backward(
    params: &SharedParams,
    (input, grad_output): (Array1<f64>, Array1<f64>),
) -> (LinearLayerParams, Array1<f64>) {
    let dw = input
        .clone()
        .insert_axis(Axis(1))
        .dot(&grad_output.clone().insert_axis(Axis(0)));
    let db = grad_output.clone();
    let da = params.borrow().weight_matrix.dot(&grad_output);

    (
        LinearLayerParams {
            weight_matrix: dw,
            bias_matrix: db,
        },
        da,
    )
}
pub fn new_linear_layer(
    in_features: usize,
    out_features: usize,
) -> (LinearLayerForward, LinearLayerBackward) {
    let scale = (2f64 / in_features as f64).sqrt();
    let weight_matrix = Array::random(
        (in_features, out_features),
        Normal::new(0.0, scale).unwrap(),
    );
    let bias_matrix = Array1::<f64>::zeros(out_features);

    let params = Rc::new(RefCell::new(LinearLayerParams {
        weight_matrix,
        bias_matrix,
    }));

    (
        CurryingMorphism::new(forward, Rc::clone(&params)),
        CurryingMorphism::new(backward, params),
    )
}
