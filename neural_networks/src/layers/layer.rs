use utils::currying_morphism::DerivatibleCurryingMorphism;

pub trait Layer {
    fn features(&self) -> (usize, usize);
}

pub trait LayerImpl<Params, Input, Output, DerivativeInput, DerivativeOutput>:
    DerivatibleCurryingMorphism<
        Params = Params,
        Input = Input,
        Output = Output,
        DerivativeInput = DerivativeInput,
        DerivativeOutput = DerivativeOutput,
    > + Layer
{
}

impl<T, Params, Input, Output, DerivativeInput, DerivativeOutput>
    LayerImpl<Params, Input, Output, DerivativeInput, DerivativeOutput> for T
where
    T: DerivatibleCurryingMorphism<
            Params = Params,
            Input = Input,
            Output = Output,
            DerivativeInput = DerivativeInput,
            DerivativeOutput = DerivativeOutput,
        > + Layer,
{
}
