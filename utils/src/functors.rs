use std::{marker::PhantomData, rc::Rc};

use crate::{
    derivative::DerivativeMorphism,
    errors::CategoryResult,
    morphism::{CurryingMorphism, Morphism},
};

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

impl<MPhism, Input, Output> DerivativeMorphism for CollectionFunctor<MPhism, Input, Output>
where
    MPhism: DerivativeMorphism + Sized + 'static,
    MPhism::Input: 'static,
    MPhism::Output: 'static,
    Input: IntoIterator<Item = MPhism::Input> + 'static,
    Output: FromIterator<MPhism::Output> + 'static,
{
    type Input = Input;
    type Output = Output;

    fn derivative(
        &self,
    ) -> std::rc::Rc<dyn Morphism<Input = Self::Input, Output = Self::Output> + 'static> {
        Rc::new(CollectionFunctor::new(self.morphism.derivative()))
    }
}

impl<MPhism, Input, Output> CurryingMorphism for CollectionFunctor<MPhism, Input, Output>
where
    MPhism: CurryingMorphism + Sized + 'static,
    MPhism::Input: 'static,
    MPhism::Output: 'static,
    Input: IntoIterator<Item = MPhism::Input> + 'static,
    Output: FromIterator<MPhism::Output> + 'static,
{
    type Params = MPhism::Params;
    type Input = Input;
    type Output = Output;

    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output> + 'static> {
        Rc::new(CollectionFunctor::new(self.morphism.curry(params)))
    }
}
