use candle_core::Device;
use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::linear::LinearLayer,
};
use std::rc::Rc;

use crate::barlow::train::BarlowTrainStep;
pub(crate) mod train;
pub fn train_barlow() -> Result<(), Box<dyn std::error::Error>> {
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
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 64))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 10))),
    ]);

    let mut projector = Sequential::new(&[
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(10, 6))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(6, 7))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(7, 10))),
    ]);
    let device = Device::cuda_if_available(0)?;
    dense.init_params(&device)?;
    let learning_rate = 0.01;
    let batch_size = 64;
    let train = BarlowTrainStep::new(dataset, learning_rate, batch_size, (5, 5), device);
    for _ in 0..10 {
        train.train(&mut dense)?;
    }

    train.test(&mut dense, testset.clone())?;
    train.export_embeddings_2d(&mut dense, &testset, 10000, "output/embeddings_2d.csv")?;
    Ok(())
}
