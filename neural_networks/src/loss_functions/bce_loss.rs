use std::rc::Rc;

use candle_core::Tensor;
use utils::{derivative::DerivativeMorphism, errors::CategoryResult, morphism::Morphism};

pub struct BCELoss;

impl Morphism for BCELoss {
    type Input = (Tensor, Tensor);
    type Output = f32;

    fn name(&self) -> &'static str {
        "BCELoss"
    }

    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        let n = al.elem_count() as f32;

        let log_al = al.log()?;
        let one_minus_al = al.affine(-1.0, 1.0)?;
        let log_one_minus_al = one_minus_al.log()?;
        let positive = (&y * &log_al)?;
        let one_minus_y = y.affine(-1.0, 1.0)?;
        let negative = (&one_minus_y * &log_one_minus_al)?;
        let loss = (&positive + &negative)?.sum_all()?.to_scalar::<f32>()?;

        Ok(-loss / n)
    }
}

pub struct DerivativeBCELoss;

impl DerivativeMorphism for BCELoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;
    type Curry = ();

    fn derivative(
        &self,
        _input: Self::Curry,
    ) -> Rc<dyn Morphism<Input = Self::Input, Output = Self::Output>> {
        Rc::new(DerivativeBCELoss)
    }
}

impl Morphism for DerivativeBCELoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn name(&self) -> &'static str {
        "DerivativeBCELoss"
    }

    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        let n = al.elem_count() as f64;
        let diff = (&al - &y)?;

        let one_minus_al = al.affine(-1.0, 1.0)?;
        let denominator = (&al * &one_minus_al)?;

        let denominator = denominator.affine(n, 0.0)?;

        Ok((&diff / &denominator)?)
    }
}
