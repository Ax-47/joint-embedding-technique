use std::{marker::PhantomData, rc::Rc};

use crate::{
    currying_morphism::CurryingMorphism,
    derivative::DerivativeMorphism,
    errors::{CategoryError, CategoryResult},
    morphism::Morphism,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn, ShapeError};

pub trait Collection {
    type Item;
    type Shape: Clone;
    type IntoElements: Iterator<Item = Self::Item>;

    fn shape(&self) -> Self::Shape;
    fn into_elements(self) -> Self::IntoElements;
}

pub trait Rebuild<A>: Sized {
    type Shape;
    fn rebuild<I>(shape: Self::Shape, items: I) -> CategoryResult<Self>
    where
        I: IntoIterator<Item = A>;
}
impl<A> Collection for Vec<A> {
    type Item = A;
    type Shape = usize;
    type IntoElements = std::vec::IntoIter<A>;

    fn shape(&self) -> usize {
        self.len()
    }
    fn into_elements(self) -> Self::IntoElements {
        self.into_iter()
    }
}

impl<A> Rebuild<A> for Vec<A> {
    type Shape = usize;
    fn rebuild<I: IntoIterator<Item = A>>(_shape: usize, items: I) -> CategoryResult<Self> {
        Ok(items.into_iter().collect())
    }
}

impl<A> Collection for Array2<A> {
    type Item = A;
    type Shape = (usize, usize);
    type IntoElements = ndarray::iter::IntoIter<A, ndarray::Ix2>;

    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    fn into_elements(self) -> Self::IntoElements {
        self.into_iter() // logical order, row-major
    }
}

impl<A> Rebuild<A> for Array2<A> {
    type Shape = (usize, usize);
    fn rebuild<I: IntoIterator<Item = A>>(shape: (usize, usize), items: I) -> CategoryResult<Self> {
        Array2::from_shape_vec(shape, items.into_iter().collect()).map_err(CategoryError::from)
    }
}
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
    Input: Collection<Item = MPhism::Input>,
    Output: Rebuild<MPhism::Output, Shape = Input::Shape>,
{
    type Input = Input;
    type Output = Output;
    fn name(&self) -> &'static str {
        "collection functor"
    }

    fn apply(&self, input: Input) -> CategoryResult<Output> {
        let shape = input.shape();
        let mapped = input
            .into_elements()
            .map(|x| self.morphism.apply(x))
            .collect::<CategoryResult<Vec<_>>>()?;
        Output::rebuild(shape, mapped)
    }
}

impl<MPhism, Input, Output> DerivativeMorphism for CollectionFunctor<MPhism, Input, Output>
where
    MPhism: DerivativeMorphism<Output = Output> + Sized + 'static,
    MPhism::Input: 'static,
    MPhism::Output: 'static,
    Input: Collection<Item = MPhism::Input> + 'static,
    Output: Rebuild<MPhism::Output, Shape = Input::Shape> + 'static,
{
    type Input = Input;
    type Curry = MPhism::Curry;
    type Output = Output;

    fn derivative(
        &self,
        input: Self::Curry,
    ) -> std::rc::Rc<dyn Morphism<Input = Self::Input, Output = Self::Output> + 'static> {
        Rc::new(CollectionFunctor::new(self.morphism.derivative(input)))
    }
}

impl<MPhism, Input, Output> CurryingMorphism for CollectionFunctor<MPhism, Input, Output>
where
    MPhism: CurryingMorphism + Sized + 'static,
    MPhism::Input: 'static,
    MPhism::Output: 'static,
    Input: Collection<Item = MPhism::Input> + 'static,
    Output: Rebuild<MPhism::Output, Shape = Input::Shape> + 'static,
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
