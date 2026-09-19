use std::rc::Rc;

use candle_core::{Device, Tensor};
use utils::{
    derivative::DerivatibleMorphism, errors::CategoryResult, monoid::Monoid, morphism::Morphism,
};

use crate::layers::{layer::LayerImpl, param_set::ParamSet};

pub type V = Tensor;

pub type DynCurrying = dyn LayerImpl<ParamSet, V, V, V, V, (ParamSet, V)>;
pub type DynDerivatible = dyn DerivatibleMorphism<V, V, V, V, (ParamSet, V)>;
pub type DynPlain = dyn DerivatibleMorphism<V, V, V, V, V>;

#[derive(Clone)]
pub enum Layer {
    CurryingMorphism(Rc<DynCurrying>),
    Morphism(Rc<DynPlain>),
}

#[derive(Clone)]
pub enum DerivativeLayer {
    CurryingMorphism(Rc<DynDerivatible>),
    Morphism(Rc<DynPlain>),
}

#[derive(Clone)]
pub enum DerivativedLayer {
    CurryingMorphism(Rc<dyn Morphism<Input = V, Output = (ParamSet, V)>>),
    Morphism(Rc<dyn Morphism<Input = V, Output = V>>),
}

impl Layer {
    pub fn curry(&self, params: &ParamSet) -> DerivativeLayer {
        match self {
            Layer::CurryingMorphism(layer) => {
                DerivativeLayer::CurryingMorphism(layer.curry(params))
            }
            Layer::Morphism(morphism) => DerivativeLayer::Morphism(morphism.clone()),
        }
    }

    pub fn is_currying_morphism(&self) -> bool {
        matches!(self, Layer::CurryingMorphism(_))
    }
}
pub struct Sequential {
    layers: Vec<Layer>,
    derivative_dense: Vec<DerivativedLayer>,
    params: Vec<ParamSet>,
}

impl Sequential {
    pub fn new(layers: &[Layer]) -> Self {
        Self {
            layers: layers.to_vec(),
            derivative_dense: Vec::with_capacity(layers.len()),
            params: Vec::with_capacity(layers.len()),
        }
    }

    pub fn init_params(&mut self, device: &Device) -> candle_core::Result<()> {
        self.params.clear();
        for layer in &self.layers {
            if let Layer::CurryingMorphism(currying) = layer {
                self.params.push(currying.init_params(device)?);
            }
        }
        Ok(())
    }

    pub fn params(&self) -> &[ParamSet] {
        &self.params
    }

    pub fn set_params(&mut self, new_params: Vec<ParamSet>) {
        debug_assert_eq!(new_params.len(), self.params.len());
        self.params = new_params;
        self.derivative_dense.clear();
    }

    pub fn forward_with_tape(
        &mut self,
        input: V,
    ) -> CategoryResult<(Tensor, Vec<DerivativedLayer>)> {
        let mut tape = Vec::with_capacity(self.layers.len());

        let mut output = input;
        let mut params_iter = self.params.iter();

        for layer in &self.layers {
            match layer {
                Layer::CurryingMorphism(currying) => {
                    let params = params_iter
                        .next()
                        .expect("not enough params for currying layers");

                    let morphism = currying.curry(params);
                    let layer_input = output;

                    tape.push(DerivativedLayer::CurryingMorphism(
                        morphism.derivative(layer_input.clone()),
                    ));

                    output = morphism.apply(layer_input)?;
                }
                Layer::Morphism(morphism) => {
                    let layer_input = output;

                    tape.push(DerivativedLayer::Morphism(
                        morphism.derivative(layer_input.clone()),
                    ));

                    output = morphism.apply(layer_input)?;
                }
            }
        }

        Ok((output, tape))
    }

    pub fn backward_with_tape(
        &mut self,
        tape: Vec<DerivativedLayer>,
        mut grad: Tensor,
    ) -> CategoryResult<Vec<ParamSet>> {
        let currying_count = tape
            .iter()
            .filter(|layer| matches!(layer, DerivativedLayer::CurryingMorphism(_)))
            .count();

        let mut gradients = Vec::with_capacity(currying_count);

        for layer in tape.into_iter().rev() {
            match layer {
                DerivativedLayer::CurryingMorphism(derivative) => {
                    let (layer_grad, input_grad) = derivative.apply(grad)?;
                    gradients.push(layer_grad);
                    grad = input_grad;
                }
                DerivativedLayer::Morphism(derivative) => {
                    grad = derivative.apply(grad)?;
                }
            }
        }

        gradients.reverse();
        Ok(gradients)
    }

    pub fn forward(&mut self, input: V) -> CategoryResult<Tensor> {
        let (output, tape) = self.forward_with_tape(input)?;
        self.derivative_dense = tape;
        Ok(output)
    }

    pub fn backward(&mut self, grad: Tensor) -> CategoryResult<Vec<ParamSet>> {
        let tapes = std::mem::take(&mut self.derivative_dense);
        self.backward_with_tape(tapes, grad)
    }
}
impl Monoid for Sequential {
    fn empty() -> Self {
        Self {
            layers: Vec::new(),
            derivative_dense: Vec::new(),
            params: Vec::new(),
        }
    }

    fn combine(&self, other: &Self) -> Self {
        let mut layers = Vec::with_capacity(self.layers.len() + other.layers.len());
        layers.extend_from_slice(&self.layers);
        layers.extend_from_slice(&other.layers);

        let mut params = Vec::with_capacity(self.params.len() + other.params.len());
        params.extend_from_slice(&self.params);
        params.extend_from_slice(&other.params);

        Self {
            layers,
            derivative_dense: Vec::new(),
            params,
        }
    }
}
