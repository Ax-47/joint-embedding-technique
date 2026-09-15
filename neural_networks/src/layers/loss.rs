use ndarray::{Array1, Array2, Axis};
use utils::{errors::CategoryResult, morphism::Morphism};

pub struct Loss;

impl Morphism for Loss {
    type Input = (Array2<f64>, Array2<f64>);
    type Output = f64;
    fn name(&self) -> &'static str {
        "Loss"
    }
    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        let diff = &al - &y;
        Ok(diff.mapv(|x| x * x).sum() / al.len() as f64)
    }
}

pub struct DerivativeLoss;
impl Morphism for DerivativeLoss {
    type Input = (Array2<f64>, Array2<f64>);
    type Output = Array2<f64>;
    fn name(&self) -> &'static str {
        "Loss"
    }
    fn apply(&self, (al, y): Self::Input) -> CategoryResult<Self::Output> {
        Ok(2.0 * (&al - &y) / al.nrows() as f64)
    }
}
