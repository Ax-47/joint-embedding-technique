use std::{marker::PhantomData, rc::Rc};

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
impl<I, O> Morphism for Rc<dyn Morphism<Input = I, Output = O> + 'static> {
    type Input = I;
    type Output = O;

    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        (**self).apply(input)
    }
}
pub struct CurryingMorphism<F, Parameter, Input, Output> {
    f: F,
    parameter: Parameter,
    _marker: PhantomData<fn(Input) -> Output>,
}
impl<F: Clone, Parameter: Clone, Input, Output> Clone
    for CurryingMorphism<F, Parameter, Input, Output>
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            parameter: self.parameter.clone(),
            _marker: PhantomData,
        }
    }
}

impl<F: std::fmt::Debug, Parameter: std::fmt::Debug, Input, Output> std::fmt::Debug
    for CurryingMorphism<F, Parameter, Input, Output>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurryingMorphism")
            .field("f", &self.f)
            .field("parameter", &self.parameter)
            .finish()
    }
}
impl<F, Parameter, Input, Output> CurryingMorphism<F, Parameter, Input, Output> {
    pub fn new(f: F, parameter: Parameter) -> Self {
        Self {
            f,
            parameter,
            _marker: PhantomData,
        }
    }
    pub fn update_params(&mut self, new_params: Parameter) {
        self.parameter = new_params;
    }
    pub fn params(&self) -> &Parameter {
        &self.parameter
    }
}

impl<F, Parameter, Input, Output> Morphism for CurryingMorphism<F, Parameter, Input, Output>
where
    F: Fn(&Parameter, Input) -> Output,
{
    type Input = Input;
    type Output = Output;
    fn name(&self) -> &'static str {
        "currying_morphism"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        Ok((self.f)(&self.parameter, input))
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
