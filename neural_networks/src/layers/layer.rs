use std::rc::Rc;

use candle_core::Device;
use utils::{currying_morphism::DerivatibleCurryingMorphism, derivative::DerivatibleMorphism};
pub trait LayerShape<Params> {
    fn features(&self) -> (usize, usize);
    fn init_params(&self, device: &Device) -> candle_core::Result<Params>;
}
pub trait LayerImpl<Params, Input, Output, Curry, DerivativeInput, DerivativeOutput>:
    DerivatibleCurryingMorphism<
        Params = Params,
        Input = Input,
        Output = Output,
        Curry = Curry,
        DerivativeInput = DerivativeInput,
        DerivativeOutput = DerivativeOutput,
    > + LayerShape<Params>
{
}

impl<T, Params, Input, Output, Curry, DerivativeInput, DerivativeOutput>
    LayerImpl<Params, Input, Output, Curry, DerivativeInput, DerivativeOutput> for T
where
    T: DerivatibleCurryingMorphism<
            Params = Params,
            Input = Input,
            Output = Output,
            Curry = Curry,
            DerivativeInput = DerivativeInput,
            DerivativeOutput = DerivativeOutput,
        > + LayerShape<Params>,
{
}
