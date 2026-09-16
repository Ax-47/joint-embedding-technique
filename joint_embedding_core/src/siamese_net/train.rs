use candle_core::Device;
use data_process::read_byte::DataSet;
use neural_networks::{
    container::sequential::Sequential,
    layers::{
        linear::LinearLayerParams,
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
pub struct TrainStep {
    dataset: DataSet,
    learning_rate: f64,
    batch_size: usize,
    device: Device,
}
impl TrainStep {
    pub fn new(dataset: DataSet, learning_rate: f64, batch_size: usize, device: Device) -> Self {
        Self {
            dataset,
            learning_rate,
            batch_size,
            device,
        }
    }
    pub fn train(&self, dense: &mut Sequential) -> CategoryResult<()> {
        for epoch in 0..10 {
            for (batch_idx, batch) in self
                .dataset
                .batch_view_iter(self.batch_size, 60_000)
                .enumerate()
            {
                let x = batch.images_tensor(&self.device)?;
                let y = batch.label_one_hot_tensor(&self.device)?;
                dense.currying();
                let (last_out, all_outs) = dense.forward_with_tape(x)?;
                let loss = Loss.apply((last_out.clone(), y.clone()))?;
                let delta = DerivativeLoss.apply((last_out, y))?;
                let grads = dense.backward(all_outs, delta)?;

                let new_params = sgd_step(&dense.params(), &grads, self.learning_rate)?;
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
}
