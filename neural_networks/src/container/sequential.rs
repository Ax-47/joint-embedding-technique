use std::rc::Rc;

use ndarray::{Array, Array1};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use utils::{
    currying_morphism::{CurryingMorphism, DerivatibleCurryingMorphism},
    derivative::{DerivatibleMorphism, DerivativeMorphism},
    monoid::Monoid,
    morphism::Morphism,
};

use crate::layers::{layer::LayerImpl, linear::LinearLayerParams};
pub type V = Array1<f64>;

pub type DynCurrying = dyn LayerImpl<LinearLayerParams, V, V, (V, V), (LinearLayerParams, V)>;

pub type DynDerivatible = dyn DerivatibleMorphism<V, V, (V, V), (LinearLayerParams, V)>;
pub type DynPlain = dyn DerivatibleMorphism<V, V, V, V>;

#[derive(Clone)]
pub enum Layer {
    CurryingMorphism(Rc<DynCurrying>),
    Morphism(Rc<DynPlain>),
}

#[derive(Clone)]
pub enum DerivativeLayer {
    CurryingMorphism(
        Rc<
            dyn Morphism<
                    Input = (Array1<f64>, Array1<f64>),
                    Output = (LinearLayerParams, Array1<f64>),
                >,
        >,
    ),
    Morphism(Rc<dyn Morphism<Input = Array1<f64>, Output = Array1<f64>>>),
}
impl Layer {
    pub fn curry(
        &self,
        params: &LinearLayerParams,
    ) -> Rc<dyn Morphism<Input = Array1<f64>, Output = Array1<f64>>> {
        match self {
            Layer::CurryingMorphism(l) => l.curry(params),
            Layer::Morphism(r) => r.clone(),
        }
    }
    pub fn is_currying_morphism(&self) -> bool {
        match self {
            Layer::CurryingMorphism(_) => true,
            Layer::Morphism(_) => false,
        }
    }
}
pub struct Sequential {
    layers: Vec<Layer>,
    dense: Vec<Rc<dyn Morphism<Input = Array1<f64>, Output = Array1<f64>>>>,
    derivative_dense: Vec<DerivativeLayer>,
    params: Vec<LinearLayerParams>,
    values: Vec<Array1<f64>>,
}
impl Sequential {
    pub fn new(layers: &[Layer]) -> Self {
        Self {
            layers: Vec::from(layers),
            dense: Vec::new(),
            params: Vec::new(),
            derivative_dense: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn init_params(&mut self) {
        let layers = std::mem::take(&mut self.layers);
        self.params.clear();
        for layer in layers.iter() {
            let param = match layer {
                Layer::CurryingMorphism(c) => {
                    let (in_features, out_features) = c.features();
                    let scale = (2f64 / in_features as f64).sqrt();
                    let weight_matrix = Array::random(
                        (in_features, out_features),
                        Normal::new(0.0, scale).unwrap(),
                    );
                    let bias_matrix = Array1::<f64>::zeros(out_features);

                    LinearLayerParams {
                        weight_matrix,
                        bias_matrix,
                    }
                }
                Layer::Morphism(r) => continue,
            };
            self.params.push(param);
        }
        self.layers = layers;
    }
    pub fn params(&self) -> Vec<LinearLayerParams> {
        self.params.clone()
    }

    pub fn set_params(&mut self, new_params: Vec<LinearLayerParams>) {
        self.params = new_params;
    }
    pub fn currying(&mut self) {
        let layers = std::mem::take(&mut self.layers);
        self.dense.clear();
        self.derivative_dense.clear();
        let mut params_iter = self.params.iter();
        for layer in layers.iter() {
            let (applied, derivative) = match layer {
                Layer::CurryingMorphism(c) => {
                    let p = params_iter
                        .next()
                        .expect("not enough params for currying layers");
                    let curried: Rc<DynDerivatible> = c.curry(p);
                    let d = curried.derivative();
                    (curried.as_morphism(), DerivativeLayer::CurryingMorphism(d))
                }
                Layer::Morphism(r) => (
                    r.clone().as_morphism(),
                    DerivativeLayer::Morphism(r.derivative()),
                ),
            };
            self.dense.push(applied);
            self.derivative_dense.push(derivative);
        }
        self.layers = layers;
    }
    pub fn forward_with_tape(
        &self,
        input: Array1<f64>,
    ) -> utils::errors::CategoryResult<(Array1<f64>, Vec<Array1<f64>>)> {
        let mut tape = vec![input.clone()];
        let mut output = input;
        for layer in self.dense.iter() {
            output = layer.apply(output)?;
            tape.push(output.clone());
        }

        Ok((output, tape))
    }

    pub fn backward(
        &self,
        tape: Vec<Array1<f64>>,
        mut grad: Array1<f64>,
    ) -> utils::errors::CategoryResult<Vec<LinearLayerParams>> {
        let mut new_params = vec![];
        for (d_layer, output) in self.derivative_dense.iter().zip(tape).rev() {
            grad = match d_layer {
                DerivativeLayer::CurryingMorphism(d_cmorphism) => {
                    let (new_p, gradd) = d_cmorphism.apply((output, grad))?;
                    new_params.push(new_p);
                    gradd
                }
                DerivativeLayer::Morphism(d_morphism) => {
                    let local = d_morphism.apply(output)?;
                    local * grad
                }
            }
        }
        new_params.reverse();
        Ok(new_params)
    }
}

impl Monoid for Sequential {
    fn empty() -> Self {
        Self {
            layers: Vec::new(),
            dense: Vec::new(),
            params: Vec::new(),
            derivative_dense: Vec::new(),
            values: Vec::new(),
        }
    }
    fn combine(&self, other: &Self) -> Self {
        let mut combined_layers = Vec::with_capacity(self.layers.len() + other.layers.len());
        combined_layers.extend(self.layers.iter().cloned());
        combined_layers.extend(other.layers.iter().cloned());

        let mut combined_values = Vec::with_capacity(self.values.len() + other.values.len());
        combined_values.extend(self.values.iter().cloned());
        combined_values.extend(other.values.iter().cloned());
        let mut combined_params = Vec::with_capacity(self.params.len() + other.params.len());
        combined_params.extend(self.params.iter().cloned());
        combined_params.extend(other.params.iter().cloned());

        Self {
            layers: combined_layers,
            dense: Vec::new(),
            derivative_dense: Vec::new(),
            params: combined_params,
            values: combined_values,
        }
    }
}
