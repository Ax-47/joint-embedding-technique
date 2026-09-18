use std::rc::Rc;

use crate::{monoid::Monoid, morphism::Morphism};

pub trait DerivativeMorphism {
    type Input;
    type Curry;
    type Output;
    fn derivative(
        &self,
        input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output> + 'static>;
}
impl<Input, Curry, Output> Clone for DerivativeMonoid<Input, Curry, Output> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
pub trait DerivatibleMorphism<I, O, C, DI, DO>:
    Morphism<Input = I, Output = O> + DerivativeMorphism<Input = DI, Curry = C, Output = DO>
{
    fn as_morphism(self: Rc<Self>) -> Rc<dyn Morphism<Input = I, Output = O>>;
}

impl<T, I, O, C, DI, DO> DerivatibleMorphism<I, O, C, DI, DO> for T
where
    T: Morphism<Input = I, Output = O>
        + DerivativeMorphism<Input = DI, Curry = C, Output = DO>
        + 'static,
{
    fn as_morphism(self: Rc<Self>) -> Rc<dyn Morphism<Input = I, Output = O>> {
        self
    }
}
pub struct DerivativeMonoid<Input, Curry, Output>(
    Vec<Rc<dyn DerivativeMorphism<Input = Input, Curry = Curry, Output = Output> + 'static>>,
);
impl<Input, Curry, Output> Monoid for DerivativeMonoid<Input, Curry, Output> {
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
