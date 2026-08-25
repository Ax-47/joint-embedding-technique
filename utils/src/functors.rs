use std::marker::PhantomData;

use crate::{errors::CategoryResult, morphism::Morphism};

#[derive(Debug, Clone)]
pub struct CollectionFunctor<MPhism, Input, Output> {
    morphism: MPhism,
    _marker: PhantomData<(Input, Output)>,
}
impl<MPhism, Input, Output> CollectionFunctor<MPhism, Input, Output> {
    pub fn new(morphism: MPhism) -> Self {
        Self {
            morphism,
            _marker: PhantomData,
        }
    }
}

impl<MPhism, Input, Output> Morphism for CollectionFunctor<MPhism, Input, Output>
where
    MPhism: Morphism,
    Input: IntoIterator<Item = MPhism::Input>,
    Output: FromIterator<MPhism::Output>,
{
    type Input = Input;
    type Output = Output;
    fn name(&self) -> &'static str {
        "collection functor"
    }

    fn apply(&self, input: Input) -> CategoryResult<Output> {
        input.into_iter().map(|x| self.morphism.apply(x)).collect()
    }
}
