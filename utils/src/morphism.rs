use std::marker::PhantomData;

use crate::errors::CategoryResult;

pub trait Morphism {
    type Input;
    type Output;
    fn name(&self) -> &'static str;
    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output>;

    fn compose<G>(self, g: G) -> Compose<Self, G>
    where
        Self: Sized,
        G: Morphism,
    {
        Compose::new(self, g)
    }
}

#[derive(Debug, Clone)]
pub struct CurryingMorphism<F, HyperParameter, Input, Output> {
    f: F,
    hyper_parameter: HyperParameter,
    _marker: PhantomData<fn(Input) -> Output>,
}

impl<F, HyperParameter, Input, Output> CurryingMorphism<F, HyperParameter, Input, Output> {
    pub fn new(f: F, hyper_parameter: HyperParameter) -> Self {
        Self {
            f,
            hyper_parameter,
            _marker: PhantomData,
        }
    }
    pub fn update_params(&mut self, new_params: HyperParameter) {
        self.hyper_parameter = new_params;
    }
    pub fn params(&self) -> &HyperParameter {
        &self.hyper_parameter
    }
}

impl<F, HyperParameter, Input, Output> Morphism
    for CurryingMorphism<F, HyperParameter, Input, Output>
where
    F: Fn(&HyperParameter, Input) -> Output,
{
    type Input = Input;
    type Output = Output;
    fn name(&self) -> &'static str {
        "currying_morphism"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        Ok((self.f)(&self.hyper_parameter, input))
    }
}
pub struct Compose<F, G> {
    f: F,
    g: G,
}

impl<F, G> Compose<F, G> {
    pub fn new(f: F, g: G) -> Self {
        Self { f, g }
    }
}

impl<F, G> Morphism for Compose<F, G>
where
    F: Morphism,
    G: Morphism<Input = F::Output>,
{
    type Input = F::Input;
    type Output = G::Output;

    fn name(&self) -> &'static str {
        "compose"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        let mid = self.f.apply(input)?;
        self.g.apply(mid)
    }
}
