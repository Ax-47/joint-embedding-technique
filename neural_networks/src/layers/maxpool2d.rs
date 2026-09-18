use std::rc::Rc;

use candle_core::{DType, Tensor};
use utils::{
    derivative::{DerivatibleMorphism, DerivativeMorphism},
    errors::CategoryResult,
    morphism::Morphism,
};

pub struct MaxPool2dLayer {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2dLayer {
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
        }
    }
}
impl Morphism for MaxPool2dLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "maxpool2d layer"
    }

    fn apply(&self, input: Tensor) -> CategoryResult<Tensor> {
        let (n, c, h, w) = input.dims4()?;
        let k = self.kernel_size;
        let s = self.stride;
        let ho = (h - k) / s + 1;
        let wo = (w - k) / s + 1;
        let windows = windows(&input, n, c, ho, wo, k, s)?;
        let out = windows.max(3)?;
        Ok(out.reshape((n, c, ho, wo))?)
    }
}
pub struct DerivativeMaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
    pub input: Tensor,
}

impl Morphism for DerivativeMaxPool2d {
    type Input = Tensor;
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "maxpool2d derivative"
    }

    fn apply(&self, grad_output: Tensor) -> CategoryResult<Tensor> {
        let (n, c, h, w) = self.input.dims4()?;
        let k = self.kernel_size;
        let s = self.stride;
        let ho = (h - k) / s + 1;
        let wo = (w - k) / s + 1;
        let kk = k * k;

        let win = windows(&self.input, n, c, ho, wo, k, s)?;
        let idx = win.argmax(3)?;

        let grad_flat = grad_output.reshape((n, c, ho * wo, 1))?;
        let idx = idx.reshape((n, c, ho * wo, 1))?;

        let arange =
            Tensor::arange(0u32, kk as u32, grad_output.device())?.reshape((1, 1, 1, kk))?;
        let mask = arange.broadcast_eq(&idx)?;

        let scattered = grad_flat.broadcast_mul(&mask.to_dtype(DType::F32)?)?;

        let grad_input = col2im(&scattered, n, c, ho, wo, k, s, h, w)?;
        Ok(grad_input)
    }
}

impl DerivativeMorphism for MaxPool2dLayer {
    type Input = Tensor;
    type Curry = Tensor;
    type Output = Tensor;

    fn derivative(&self, input: Self::Curry) -> Rc<dyn Morphism<Input = Tensor, Output = Tensor>> {
        Rc::new(DerivativeMaxPool2d {
            kernel_size: self.kernel_size,
            stride: self.stride,
            input,
        })
    }
}
fn windows(
    input: &Tensor,
    n: usize,
    c: usize,
    ho: usize,
    wo: usize,
    k: usize,
    _s: usize,
) -> CategoryResult<Tensor> {
    let x = input
        .reshape((n, c, ho, k, wo, k))?
        .permute((0, 1, 2, 4, 3, 5))?
        .reshape((n, c, ho * wo, k * k))?;
    Ok(x)
}

fn col2im(
    cols: &Tensor,
    n: usize,
    c: usize,
    ho: usize,
    wo: usize,
    k: usize,
    _s: usize,
    h: usize,
    w: usize,
) -> CategoryResult<Tensor> {
    let x = cols
        .reshape((n, c, ho, wo, k, k))?
        .permute((0, 1, 2, 4, 3, 5))?
        .reshape((n, c, h, w))?;
    Ok(x)
}
