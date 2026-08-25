use std::marker::PhantomData;

use crate::errors::CategoryResult;

pub trait Morphism<Input, Output> {
    fn name(&self) -> &'static str;
    fn apply(&self, input: Input) -> CategoryResult<Output>;

    fn compose<G, OutputG>(self, g: G) -> Compose<Self, G, Output>
    where
        Self: Sized,
        G: Morphism<Output, OutputG>,
    {
        Compose::new(self, g)
    }
}

#[derive(Debug, Clone)]
pub struct CurryingMorphism<F, HyperParameter> {
    f: F,
    hyper_parameter: HyperParameter,
}

impl<F, HyperParameter> CurryingMorphism<F, HyperParameter> {
    pub fn new(f: F, hyper_parameter: HyperParameter) -> Self {
        Self { f, hyper_parameter }
    }
}

impl<F, HyperParameter, Input, Output> Morphism<Input, Output>
    for CurryingMorphism<F, HyperParameter>
where
    F: Fn(&HyperParameter, Input) -> Output,
{
    fn name(&self) -> &'static str {
        "currying_morphism"
    }

    fn apply(&self, input: Input) -> CategoryResult<Output> {
        Ok((self.f)(&self.hyper_parameter, input))
    }
}
#[derive(Debug, Clone)]
pub struct Compose<F, G, Middle> {
    first: F,
    second: G,
    _middle: PhantomData<Middle>,
}

impl<F, G, Middle> Compose<F, G, Middle> {
    pub fn new(first: F, second: G) -> Self {
        Self {
            first,
            second,
            _middle: PhantomData,
        }
    }
}
impl<Input, Middle, Output, F, G> Morphism<Input, Output> for Compose<F, G, Middle>
where
    F: Morphism<Input, Middle>,
    G: Morphism<Middle, Output>,
{
    fn name(&self) -> &'static str {
        "composition"
    }

    fn apply(&self, input: Input) -> CategoryResult<Output> {
        let middle = self.first.apply(input)?;
        self.second.apply(middle)
    }
}
