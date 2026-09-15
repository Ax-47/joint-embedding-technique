use std::rc::Rc;

use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::{
        linear::{LinearLayer, LinearLayerParams},
        loss::{DerivativeLoss, Loss},
    },
};
use utils::{functors::CollectionFunctor, morphism::Morphism};
fn sgd_step(
    params: &[LinearLayerParams],
    grads: &[LinearLayerParams],
    lr: f64,
) -> Vec<LinearLayerParams> {
    let mut new_params = Vec::with_capacity(params.len() + grads.len());
    for (param, grad) in params.iter().zip(grads) {
        new_params.push(LinearLayerParams {
            weight_matrix: &param.weight_matrix - lr * &grad.weight_matrix,
            bias_matrix: &param.bias_matrix - lr * &grad.bias_matrix,
        });
    }
    new_params
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let l1 = Rc::new(LinearLayer::new(784, 64));
    let l2 = Rc::new(LinearLayer::new(64, 64));
    let l3 = Rc::new(LinearLayer::new(64, 10));
    let relu = Rc::new(CollectionFunctor::new(Relu));
    let mut dense = Sequential::new(&[
        Layer::CurryingMorphism(l1.clone()),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(l2.clone()),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(l3.clone()),
    ]);
    dense.init_params();
    let learning_rate = 0.01;
    for epoch in 0..10 {
        for (batch_idx, batch) in dataset.batch_view_iter(1, 60_000).enumerate() {
            let x = batch.images_vecf64().into_flat();
            let y = batch.label_one_hot().into_flat();

            dense.currying();
            let (last_out, all_outs) = dense.forward_with_tape(x.clone())?;
            let loss = Loss.apply((last_out.clone(), y.clone()))?;
            let delta = DerivativeLoss.apply((last_out, y))?;
            let grads = dense.backward(all_outs, delta)?;

            let new_params = sgd_step(&dense.params(), &grads, learning_rate);
            dense.set_params(new_params);

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
