use utils::{
    activation_function::ActivationFunctionPair, errors::CategoryResult, morphism::Morphism,
};

use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

pub struct LinearLayer {
    activation_function: ActivationFunctionPair,
    weight_matrix: Array2<f64>,
    bias_matrix: Array1<f64>,
}

impl LinearLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        activation_function: ActivationFunctionPair,
    ) -> Self {
        let scale = (2f64 / in_features as f64).sqrt();
        let weight_matrix = Array::random(
            (in_features, out_features),
            Normal::new(0.0, scale).unwrap(),
        );
        let bias_matrix = Array1::<f64>::zeros(out_features);
        Self {
            activation_function,
            weight_matrix,
            bias_matrix,
        }
    }
}
type Input = Array1<f64>;
type Output = Array1<f64>;
impl Morphism<Input, Output> for LinearLayer {
    fn name(&self) -> &'static str {
        "LinearLayer"
    }
    fn apply(&self, input: Input) -> CategoryResult<Array1<f64>> {
        Ok((&self.weight_matrix.t().dot(&input) - &self.bias_matrix)
            .mapv_into(&self.activation_function.function))
    }
}
