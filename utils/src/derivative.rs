use std::rc::Rc;

use crate::{monoid::Monoid, morphism::Morphism};

pub trait DerivativeMorphism {
    type Input;
    type Output;
    fn derivative(&self) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output> + 'static>;
}
impl<Input, Output> Clone for DerivativeMonoid<Input, Output> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
pub struct DerivativeMonoid<Input, Output>(
    Vec<Rc<dyn DerivativeMorphism<Input = Input, Output = Output> + 'static>>,
);
impl<Input, Output> Monoid for DerivativeMonoid<Input, Output> {
    fn empty() -> Self {
        Self(Vec::new())
    }
    fn combine(&self, other: &Self) -> Self {
        let mut combined = Vec::with_capacity(self.0.len() + other.0.len());
        combined.extend(other.0.iter().cloned());
        combined.extend(self.0.iter().cloned());
        Self(combined)
    }
}
