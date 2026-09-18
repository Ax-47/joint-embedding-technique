use utils::currying_morphism::DerivatibleCurryingMorphism;

pub trait Layer {
    fn features(&self) -> (usize, usize);
}

pub trait LayerImpl<Params, Input, Output, Curry, DerivativeInput, DerivativeOutput>:
    DerivatibleCurryingMorphism<
        Params = Params,
        Input = Input,
        Curry = Curry,
        Output = Output,
        DerivativeInput = DerivativeInput,
        DerivativeOutput = DerivativeOutput,
    > + Layer
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
        > + Layer,
{
}
