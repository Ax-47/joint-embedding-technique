use ndarray::{Array1, Array2};
use utils::activation_function::ActivationFunctionPair;
pub struct CNNModel<const N: usize> {
    activation_function: ActivationFunctionPair,
    learning_rate: f64,
    layer: [u64; N],
    bias_matrix: [Array1<f64>; N],
    weight_matrix: [Array2<f64>; N],
}
impl<const N: usize> CNNModel<N> {
    fn forward_propagation(&self, layer: usize, previous: &Array1<f64>) -> Array1<f64> {
        (self.weight_matrix[layer].dot(previous) - &self.bias_matrix[layer])
            .mapv_into(&self.activation_function.function)
    }
    fn backward_propagation(
        &self,
        layer: usize,
        previous: &Array1<f64>,
        current: &Array1<f64>,
        next: &Array1<f64>,
    ) -> Array2<f64> {
        let delta = 2f64 * (current - next);
        unimplemented!()
    }
}
