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
