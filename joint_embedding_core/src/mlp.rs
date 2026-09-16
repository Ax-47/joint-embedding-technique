use candle_core::Device;
use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::linear::LinearLayer,
};
use std::rc::Rc;

use crate::mlp::train::TrainStep;
pub(crate) mod train;
pub fn train_mlp() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let testset = DataSet::new(
        "MNIST/t10k-images-idx3-ubyte",
        "MNIST/t10k-labels-idx1-ubyte",
    )?;
    let relu = Rc::new(Relu);
    let mut dense = Sequential::new(&[
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(784, 64))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 64))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 10))),
    ]);
    let device = Device::cuda_if_available(0)?;
    dense.init_params(device.clone())?;
    let learning_rate = 0.01;
    let batch_size = 64;
    let train = TrainStep::new(dataset, learning_rate, batch_size, device);
    train.train(&mut dense)?;
    train.test(&dense, testset)?;
    Ok(())
}
