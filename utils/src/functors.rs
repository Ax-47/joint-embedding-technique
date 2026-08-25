use std::marker::PhantomData;

use crate::{errors::CategoryResult, morphism::Morphism};

#[derive(Debug, Clone)]
pub struct CollectionFunctor<MPhism, A, B> {
    morphism: MPhism,
    _type: PhantomData<(A, B)>,
}
impl<MPhism, A, B> CollectionFunctor<MPhism, A, B> {
    pub fn new(morphism: MPhism) -> Self {
        Self {
            morphism,
            _type: PhantomData,
        }
    }
}

impl<MPhism, Input, Output, A, B> Morphism<Input, Output> for CollectionFunctor<MPhism, A, B>
where
    MPhism: Morphism<A, B>,
    Input: IntoIterator<Item = A>,
    Output: FromIterator<B>,
{
    fn name(&self) -> &'static str {
        "collection functor"
    }

    fn apply(&self, input: Input) -> CategoryResult<Output> {
        input.into_iter().map(|x| self.morphism.apply(x)).collect()
    }
}
