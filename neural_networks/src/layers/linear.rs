use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use std::rc::Rc;
use utils::currying_morphism::DerivatibleCurryingMorphism;
use utils::derivative::DerivatibleMorphism;
use utils::derivative::DerivativeMorphism;
use utils::errors::CategoryResult;
use utils::morphism::Morphism;

use crate::layers::layer::LayerShape;
use crate::layers::param_set::ParamSet;
#[derive(Debug, Clone)]
pub struct LinearLayerParams {
    pub weight_matrix: Tensor,
    pub bias_matrix: Tensor,
}

pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
        }
    }
}
impl LayerShape<ParamSet> for LinearLayer {
    fn features(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }

    fn init_params(&self, device: &Device) -> candle_core::Result<ParamSet> {
        let bound = (1.0 / self.in_features as f32).sqrt();
        let weight = Tensor::rand(-bound, bound, (self.in_features, self.out_features), device)?;
        let bias = Tensor::zeros(&[self.out_features], DType::F32, device)?;
        Ok(ParamSet::new([weight, bias]))
    }
}
pub struct AppliedLinearLayer {
    params: ParamSet,
}

impl DerivatibleCurryingMorphism for LinearLayer {
    type Params = ParamSet;
    type Input = Tensor;
    type Output = Tensor;
    type Curry = Tensor;
    type DerivativeInput = Tensor;
    type DerivativeOutput = (ParamSet, Tensor);

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
    > {
        Rc::new(AppliedLinearLayer {
            params: params.clone(),
        })
    }
}

impl Morphism for AppliedLinearLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "linear layer"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        let weight = self.params.get(0);
        let bias = self.params.get(1);
        let output = input.matmul(weight)?;
        Ok(output.broadcast_add(bias)?)
    }
}
impl DerivativeMorphism for AppliedLinearLayer {
    type Input = Tensor;
    type Curry = Tensor;
    type Output = (ParamSet, Tensor);

    fn derivative(
        &self,
        input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeLinearLayer {
            params: self.params.clone(),
            input,
        })
    }
}
pub struct DerivativeLinearLayer {
    pub params: ParamSet,
    pub input: Tensor,
}

impl Morphism for DerivativeLinearLayer {
    type Input = Tensor;
    type Output = (ParamSet, Tensor);

    fn name(&self) -> &'static str {
        "linear layer derivative"
    }

    fn apply(&self, grad_output: Self::Input) -> CategoryResult<Self::Output> {
        let weight = self.params.get(0);
        let grad_weight = self.input.t()?.matmul(&grad_output)?;
        let grad_bias = grad_output.sum(0)?;
        let grad_input = grad_output.matmul(&weight.t()?)?;
        let grads = ParamSet::new([grad_weight, grad_bias]);
        Ok((grads, grad_input))
    }
}
