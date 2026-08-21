pub type ActivationFunctionType = Box<dyn Fn(f64) -> f64>;
pub struct ActivationFunctionPair {
    pub function: ActivationFunctionType,
    pub derivative_function: ActivationFunctionType,
}
impl ActivationFunctionPair {
    pub fn relu() -> Self {
        Self {
            function: Box::new(relu),
            derivative_function: Box::new(relu_prime),
        }
    }

    pub fn leaky_relu(alpha: f64) -> Self {
        Self {
            function: Box::new(move |x: f64| leaky_relu(alpha, x)),
            derivative_function: Box::new(move |x: f64| leaky_relu_prime(alpha, x)),
        }
    }
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

pub fn relu_prime(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 }
}

pub fn leaky_relu(alpha: f64, x: f64) -> f64 {
    if x < 0.0 { alpha * x } else { x }
}

pub fn leaky_relu_prime(alpha: f64, x: f64) -> f64 {
    if x < 0.0 { alpha } else { 1.0 }
}
