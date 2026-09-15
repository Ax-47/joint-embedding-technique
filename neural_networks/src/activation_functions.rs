use core::f64;
use ndarray::Array1;
use std::rc::Rc;
use utils::{
    currying_morphism::CurryingMorphism, derivative::DerivativeMorphism, errors::CategoryResult,
    functors::CollectionFunctor, morphism::Morphism,
};
pub struct Relu;

impl Morphism for Relu {
    type Input = f64;
    type Output = f64;
    fn name(&self) -> &'static str {
        "relu"
    }
    fn apply(&self, x: f64) -> CategoryResult<f64> {
        Ok(relu(x))
    }
}

pub struct ReluPrime;

impl Morphism for ReluPrime {
    type Input = f64;
    type Output = f64;
    fn name(&self) -> &'static str {
        "relu"
    }
    fn apply(&self, x: f64) -> CategoryResult<f64> {
        Ok(relu_prime(x))
    }
}

impl DerivativeMorphism for Relu {
    type Input = f64;
    type Output = f64;
    fn derivative(&self) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(ReluPrime)
    }
}

impl CurryingMorphism for Relu {
    type Params = ();
    type Input = f64;
    type Output = f64;
    fn curry(
        &self,
        _params: &Self::Params,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(Relu)
    }
}
pub struct LeakyRelu {
    alpha: f64,
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
