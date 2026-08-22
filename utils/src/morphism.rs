use std::marker::PhantomData;

use crate::errors::CategoryResult;

pub trait Morphism<Input, Output> {
    fn name(&self) -> &'static str;
    fn apply(&self, input: Input) -> CategoryResult<Output>;
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
