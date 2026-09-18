use std::rc::Rc;

use candle_core::{DType, Device, Tensor};

use utils::{
    currying_morphism::DerivatibleCurryingMorphism,
    derivative::{DerivatibleMorphism, DerivativeMorphism},
    errors::CategoryResult,
    morphism::Morphism,
};

use crate::layers::{layer::LayerShape, param_set::ParamSet};

#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dConfig {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups: 1,
        }
    }
}

pub struct Conv2dLayer {
    pub config: Conv2dConfig,
}

impl Conv2dLayer {
    pub fn new(config: Conv2dConfig) -> Self {
        Self { config }
    }
}

impl LayerShape<ParamSet> for Conv2dLayer {
    fn features(&self) -> (usize, usize) {
        (self.config.in_channels, self.config.out_channels)
    }

    fn init_params(&self, device: &Device) -> candle_core::Result<ParamSet> {
        let k = self.config.kernel_size;
        let fan_in = (self.config.in_channels / self.config.groups) * k * k;
        let std = (2.0 / fan_in as f64).sqrt();
        let weight = Tensor::randn(
            0.0,
            std,
            &[
                self.config.out_channels,
                self.config.in_channels / self.config.groups,
                k,
                k,
            ],
            device,
        )?
        .to_dtype(DType::F32)?;

        let bias = Tensor::zeros(&[self.config.out_channels], DType::F32, device)?;

        Ok(ParamSet::new([weight, bias]))
    }
}
pub struct AppliedConv2dLayer {
    pub params: ParamSet,
    pub config: Conv2dConfig,
}

impl DerivatibleCurryingMorphism for Conv2dLayer {
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
        Rc::new(AppliedConv2dLayer {
            params: params.clone(),
            config: self.config,
        })
    }
}

impl Morphism for AppliedConv2dLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "conv2d layer (applied)"
    }

    fn apply(&self, input: Self::Input) -> CategoryResult<Self::Output> {
        let weight = self.params.get(0);
        let bias = self.params.get(1);

        let y = input.conv2d(
            weight,
            self.config.padding,
            self.config.stride,
            1,
            self.config.groups,
        )?;

        let bias = bias.reshape((1, self.config.out_channels, 1, 1))?;
        Ok(y.broadcast_add(&bias)?)
    }
}
pub struct DerivativeConv2dLayer {
    pub params: ParamSet,
    pub config: Conv2dConfig,
    pub input: Tensor,
}

impl Morphism for DerivativeConv2dLayer {
    type Input = Tensor;
    type Output = (ParamSet, Tensor);

    fn name(&self) -> &'static str {
        "conv2d layer (derivative)"
    }

    fn apply(&self, grad_output: Self::Input) -> CategoryResult<Self::Output> {
        let weight = self.params.get(0);

        let db = grad_output.sum(0)?.sum(1)?.sum(1)?;

        let input_t = self.input.transpose(0, 1)?;
        let grad_out_t = grad_output.transpose(0, 1)?;

        let dw = grad_out_t.conv2d(
            &input_t,
            self.config.padding,
            self.config.stride,
            1,
            self.config.groups,
        )?;

        let da =
            grad_output.conv_transpose2d(weight, self.config.padding, 0, self.config.stride, 1)?;

        let grads = ParamSet::new([dw, db]);
        Ok((grads, da))
    }
}

impl DerivativeMorphism for AppliedConv2dLayer {
    type Input = Tensor;
    type Curry = Tensor;
    type Output = (ParamSet, Tensor);

    fn derivative(
        &self,
        input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeConv2dLayer {
            params: self.params.clone(),
            config: self.config,
            input,
        })
    }
}
