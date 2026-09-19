use candle_core::{DType, Device, Tensor};
use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::{
        conv2d::{Conv2dConfig, Conv2dLayer},
        flatten::FlattenLayer,
        linear::LinearLayer,
        maxpool2d::MaxPool2dLayer,
        param_set::ParamSet,
    },
};
use std::{path::Path, rc::Rc};
use utils::errors::CategoryResult;

pub(crate) mod train;
use crate::cnn::train::TrainStep;

pub fn save_params_npy(params: &[ParamSet], output_dir: &str) -> CategoryResult<()> {
    let dir = Path::new(output_dir);
    std::fs::create_dir_all(dir)?;

    for (layer_index, ps) in params.iter().enumerate() {
        ps.save_npy(dir, layer_index)?;
    }

    println!("บันทึก parameters {} layers ไปที่ {}", params.len(), output_dir);
    Ok(())
}
pub fn train_cnn() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let testset = DataSet::new(
        "MNIST/t10k-images-idx3-ubyte",
        "MNIST/t10k-labels-idx1-ubyte",
    )?;

    let relu = Rc::new(Relu);

    let mut net = Sequential::new(&[
        // Block 1: Conv 1 -> 32 (5x5, pad 2 เพื่อรักษาขนาด 28x28) + Pool -> 14x14
        Layer::CurryingMorphism(Rc::new(Conv2dLayer::new(Conv2dConfig::new(1, 32, 5, 1, 2)))),
        Layer::Morphism(relu.clone()),
        Layer::Morphism(Rc::new(MaxPool2dLayer::new(2))),
        // Block 2: Conv 32 -> 64 (3x3, pad 1) + Pool -> 7x7
        Layer::CurryingMorphism(Rc::new(Conv2dLayer::new(Conv2dConfig::new(
            32, 64, 3, 1, 1,
        )))),
        Layer::Morphism(relu.clone()),
        Layer::Morphism(Rc::new(MaxPool2dLayer::new(2))),
        // Block 3: Conv 64 -> 128 (3x3, pad 1) ซ้อนกัน 2 ชั้นแบบ AlexNet
        Layer::CurryingMorphism(Rc::new(Conv2dLayer::new(Conv2dConfig::new(
            64, 128, 3, 1, 1,
        )))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(Conv2dLayer::new(Conv2dConfig::new(
            128, 128, 3, 1, 1,
        )))),
        Layer::Morphism(relu.clone()),
        // Head: Flatten (128 * 7 * 7 = 6,272) -> FC 256 -> FC 10
        Layer::Morphism(Rc::new(FlattenLayer)),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(128 * 7 * 7, 256))),
        Layer::Morphism(relu.clone()),
        Layer::CurryingMorphism(Rc::new(LinearLayer::new(256, 10))),
    ]);

    let device = Device::cuda_if_available(0)?;
    net.init_params(&device)?;
    let learning_rate = 0.001;
    let batch_size = 64;

    let mut train = TrainStep::new(dataset, learning_rate, batch_size, device);
    for _ in 0..10 {
        train.train(&mut net)?;
    }
    save_params_npy(net.params(), "output/params_cnn")?;
    train.test(&mut net, testset)?;

    Ok(())
}
