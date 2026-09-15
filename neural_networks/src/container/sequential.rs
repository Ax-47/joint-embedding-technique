use std::rc::Rc;

use ndarray::Array1;
use utils::{
    monoid::Monoid,
    morphism::{CurryingMorphism, Morphism},
};

use crate::layers::linear::LinearLayerParams;
#[derive(Clone)]
pub enum Layer {
    CurryingMorphism(
        Rc<
            dyn CurryingMorphism<
                    Params = LinearLayerParams,
                    Input = Array1<f64>,
                    Output = Array1<f64>,
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
    params: Vec<LinearLayerParams>,
    values: Vec<Array1<f64>>,
}
impl Sequential {
    pub fn new(layers: &[Layer]) -> Self {
        Self {
            layers: Vec::from(layers),
            dense: Vec::new(),
            params: Vec::new(),
            values: Vec::new(),
        }
    }
    pub fn currying(&mut self, params: &[LinearLayerParams]) {
        let layers = std::mem::take(&mut self.layers);
        self.dense.clear();
        let mut params_iter = params.iter();
        for layer in layers.iter() {
            let applied = match layer {
                Layer::CurryingMorphism(c) => {
                    let p = params_iter
                        .next()
                        .expect("not enough params for currying layers");
                    c.curry(p)
                }
                Layer::Morphism(r) => r.clone(),
            };
            self.dense.push(applied);
        }
        self.layers = layers;
        self.params = params.to_vec();
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
}

impl Monoid for Sequential {
    fn empty() -> Self {
        Self {
            layers: Vec::new(),
            dense: Vec::new(),
            params: Vec::new(),
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
            params: combined_params,
            values: combined_values,
        }
    }
}
