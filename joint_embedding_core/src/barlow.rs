use candle_core::Device;
use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::linear::LinearLayer,
};
use std::rc::Rc;
use utils::monoid::Monoid;

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

    let encoder = Sequential::new(&[
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(784, 128))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(128, 128))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(128, 64))),
    ]);

    // Projector:
    // [N, 64] -> [N, 128] -> [N, 128] -> [N, 64]
    let projector = Sequential::new(&[
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(64, 128))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(128, 128))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(128, 64))),
    ]);
    let mut dense = encoder.combine(&projector);
    let device = Device::cuda_if_available(0)?;
    dense.init_params(&device)?;
    let learning_rate = 0.001;
    let batch_size = 64;
    let train = BarlowTrainStep::new(dataset, learning_rate, batch_size, (5, 5), device);
    train.train(&mut dense)?;

    train.test(&mut dense, testset.clone())?;
    train.export_embeddings_2d(&mut dense, &testset, 10000, "output/embeddings_2d.csv")?;
    Ok(())
}
