use std::rc::Rc;

use candle_core::{DType, Device, Tensor};

use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::{
        linear::{LinearLayer, LinearLayerParams},
        loss::{DerivativeLoss, Loss},
    },
};
use utils::{errors::CategoryResult, morphism::Morphism};
fn sgd_step(
    params: &[LinearLayerParams],
    grads: &[LinearLayerParams],
    lr: f64,
) -> CategoryResult<Vec<LinearLayerParams>> {
    let mut new_params = Vec::with_capacity(params.len() + grads.len());
    for (param, grad) in params.iter().zip(grads) {
        new_params.push(LinearLayerParams {
            weight_matrix: (&param.weight_matrix - (lr * &grad.weight_matrix)?)?,
            bias_matrix: (&param.bias_matrix - (lr * &grad.bias_matrix)?)?,
        });
    }
    Ok(new_params)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let relu = Rc::new(Relu);

    let mut dense = Sequential::new(&[
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(784, 64))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 64))),
        Layer::Morphism(relu),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 10))),
    ]);
    let device = Device::cuda_if_available(0)?;
    dense.init_params(device.clone())?;
    let learning_rate = 0.01;
    let batch_size = 64;
    for epoch in 0..10 {
        for (batch_idx, batch) in dataset.batch_view_iter(batch_size, 60_000).enumerate() {
            let x = batch.images_tensor(&device)?;
            let y = batch.label_one_hot_tensor(&device)?;
            dense.currying();
            let (last_out, all_outs) = dense.forward_with_tape(x)?;
            let loss = Loss.apply((last_out.clone(), y.clone()))?;
            let delta = DerivativeLoss.apply((last_out, y))?;
            let grads = dense.backward(all_outs, delta)?;

            let new_params = sgd_step(&dense.params(), &grads, learning_rate)?;
            dense.set_params(new_params);

            if batch_idx % 100 == 0 {
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
