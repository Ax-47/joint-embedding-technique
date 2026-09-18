use std::rc::Rc;

use crate::{derivative::DerivatibleMorphism, morphism::Morphism};

pub trait CurryingMorphism {
    type Params;
    type Input;
    type Output;
    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>>;
}

pub trait DerivatibleCurryingMorphism {
    type Params;
    type Input;
    type Curry;
    type Output;

    type DerivativeInput;
    type DerivativeOutput;
    fn curry(
        &self,
        params: &Self::Params,
    ) -> Rc<
        dyn DerivatibleMorphism<
                Self::Input,
                Self::Output,
                Self::Curry,
                Self::DerivativeInput,
                Self::DerivativeOutput,
            >,
    >;
}
