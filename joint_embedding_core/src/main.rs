use std::{cell::RefCell, rc::Rc};

use data_process::read_byte::DataSet;
use ndarray::Array1;
use neural_networks::{
    activation_functions::{Relu, ReluPrime, relu},
    layers::{
        linear::{LinearLayerParams, new_linear_layer},
        loss::{DerivativeLoss, Loss},
    },
};
use utils::{
    functors::CollectionFunctor,
    morphism::{self, Morphism},
};
fn sgd_step(params: &Rc<RefCell<LinearLayerParams>>, grad: &LinearLayerParams, lr: f64) {
    let (new_w, new_b) = {
        let p = params.borrow();
        (
            &p.weight_matrix - lr * &grad.weight_matrix,
            &p.bias_matrix - lr * &grad.bias_matrix,
        )
    };
    let mut p = params.borrow_mut();
    p.weight_matrix = new_w;
    p.bias_matrix = new_b;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let (forward1, backward1) = new_linear_layer(784, 64);
    let (forward2, backward2) = new_linear_layer(64, 64);
    let (forward3, backward3) = new_linear_layer(64, 10);
    let relu = CollectionFunctor::new(Relu);
    let relu_prime_layer = CollectionFunctor::new(ReluPrime);
    let learning_rate = 0.01;
    for epoch in 0..10 {
        for (batch_idx, batch) in dataset.batch_view_iter(1, 60_000).enumerate() {
            let x = batch.images_vecf64().into_flat();
            let y = batch.label_one_hot().into_flat();

            let z1 = forward1.apply(x.clone())?;
            let a1: Array1<f64> = relu.apply(z1.clone())?;
            let z2 = forward2.apply(a1.clone())?;
            let a2: Array1<f64> = relu.apply(z2.clone())?;
            let a3 = forward3.apply(a2.clone())?;

            let loss = Loss.apply((a3.clone(), y.clone()))?;
            let da3 = DerivativeLoss.apply((a3, y))?;

            let (grad3, da2_raw) = backward3.apply((a2, da3))?;
            let dz2: Array1<f64> = relu_prime_layer.apply(z2)?;
            let da2 = &da2_raw * &dz2;

            let (grad2, da1_raw) = backward2.apply((a1, da2))?;
            let dz1 = relu_prime_layer.apply(z1)?;
            let da1 = &da1_raw * &dz1;

            let (grad1, _) = backward1.apply((x, da1))?;

            sgd_step(forward1.params(), &grad1, learning_rate);
            sgd_step(forward2.params(), &grad2, learning_rate);
            sgd_step(forward3.params(), &grad3, learning_rate);

            if batch_idx % 500 == 0 {
                println!(
                    "epoch: {}/10, batch: {}, loss: {}",
                    epoch + 1,
                    batch_idx,
                    loss
                );
            }
        }
    }
    Ok(())
}
