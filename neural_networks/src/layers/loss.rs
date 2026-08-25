use ndarray::{Array1, Array2};
use utils::{errors::CategoryResult, morphism::Morphism};

pub struct Loss;

impl Morphism for Loss {
    type Input = (Array1<f64>, Array1<f64>);
    type Output = f64;
    fn name(&self) -> &'static str {
        "Loss"
    }
    fn apply(&self, (al, y): (Array1<f64>, Array1<f64>)) -> CategoryResult<f64> {
        let cost = &al - &y;
        Ok(cost.pow2().sum())
    }
}

pub struct DerivativeLoss;
impl Morphism for DerivativeLoss {
    type Input = (Array1<f64>, Array1<f64>);
    type Output = Array1<f64>;
    fn name(&self) -> &'static str {
        "Loss"
    }
    fn apply(&self, (al, y): (Array1<f64>, Array1<f64>)) -> CategoryResult<Array1<f64>> {
        let cost = 2f64 * (&al - &y);
        Ok(cost)
    }
}
