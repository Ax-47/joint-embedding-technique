use candle_core::Device;
use data_process::read_byte::DataSet;
use neural_networks::{
    activation_functions::Relu,
    container::sequential::{Layer, Sequential},
    layers::linear::LinearLayer,
};
use std::rc::Rc;

use crate::mlp::{train::TrainStep, train_mlp};

mod mlp;
mod siamese_net;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    train_mlp()?;
    Ok(())
}
