use utils::morphism::CurryingMorphism;

use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    weight_matrix: Array2<f64>,
    bias_matrix: Array1<f64>,
}
type LinearLayerFn = fn(&LinearLayerParams, Array1<f64>) -> Array1<f64>;
pub type LinearLayer = CurryingMorphism<LinearLayerFn, LinearLayerParams>;
fn forward(params: &LinearLayerParams, input: Array1<f64>) -> Array1<f64> {
    params.weight_matrix.t().dot(&input) + &params.bias_matrix
}
pub fn new_linear_layer(in_features: usize, out_features: usize) -> LinearLayer {
    let scale = (2f64 / in_features as f64).sqrt();
    let weight_matrix = Array::random(
        (in_features, out_features),
        Normal::new(0.0, scale).unwrap(),
    );
    let bias_matrix = Array1::<f64>::zeros(out_features);
    let params = LinearLayerParams {
        weight_matrix,
        bias_matrix,
    };

    CurryingMorphism::new(forward, params)
}
