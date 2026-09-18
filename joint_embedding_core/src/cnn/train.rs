use candle_core::{DType, Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::{
    container::sequential::Sequential,
    layers::{
        linear::LinearLayerParams,
        loss::{DerivativeLoss, Loss},
        param_set::ParamSet,
    },
};
use utils::{
    errors::{CategoryError, CategoryResult},
    morphism::Morphism,
};

pub fn sgd_all(
    params: &[ParamSet],
    grads: &[ParamSet],
    learning_rate: f64,
) -> CategoryResult<Vec<ParamSet>> {
    if params.len() != grads.len() {
        return Err(CategoryError::InvalidInput(format!(
            "params/grads layer count mismatch: {} vs {}",
            params.len(),
            grads.len()
        )));
    }

    params
        .iter()
        .zip(grads.iter())
        .map(|(param, grad)| param.sgd(grad, learning_rate).map_err(CategoryError::from))
        .collect()
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

    /// MNIST:
    /// [N, 784] -> [N, 1, 28, 28]
    fn images_for_cnn(&self, images: Tensor) -> candle_core::Result<Tensor> {
        let images = images.to_dtype(DType::F32)?;
        let batch_size = images.dim(0)?;

        if images.dims().len() == 4 {
            return Ok(images);
        }

        images.reshape((batch_size, 1, 28, 28))
    }

    pub fn train(&self, network: &mut Sequential) -> CategoryResult<()> {
        for epoch in 0..10 {
            for (batch_idx, batch) in self
                .dataset
                .batch_view_iter(self.batch_size, self.dataset.labels.len())
                .enumerate()
            {
                let images = batch.images_tensor(&self.device)?;
                let images = self.images_for_cnn(images)?;

                let targets = batch.label_one_hot_tensor(&self.device)?;

                // logits: [N, 10]
                let logits = network.forward(images)?;

                let loss = Loss.apply((logits.clone(), targets.clone()))?;

                let grad_logits = DerivativeLoss.apply((logits, targets))?;

                let grads = network.backward(grad_logits)?;

                let new_params = sgd_all(network.params(), &grads, self.learning_rate)?;

                network.set_params(new_params);

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

    pub fn test(&self, network: &mut Sequential, testset: DataSet) -> CategoryResult<()> {
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for (batch_idx, batch) in testset
            .batch_view_iter(100, testset.labels.len())
            .enumerate()
        {
            let images = batch.images_tensor(&self.device)?;
            let images = self.images_for_cnn(images)?;

            let labels = batch.labels_u32_tensor(&self.device)?;

            // logits: [N, 10]
            let logits = network.forward(images)?;

            // predicted: [N]
            let predicted = logits.argmax(1)?;

            let matches = predicted.eq(&labels)?;

            let correct = matches
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()? as usize;

            let batch_size = labels.dim(0)?;

            total_correct += correct;
            total_samples += batch_size;

            println!("batch:           {}", batch_idx);
            println!("correct (batch): {}", correct);
        }

        if total_samples == 0 {
            return Err(CategoryError::InvalidInput("test set is empty".to_string()));
        }

        let accuracy = total_correct as f64 / total_samples as f64 * 100.0;

        println!("Accuracy = {:.2}%", accuracy);

        Ok(())
    }
}
