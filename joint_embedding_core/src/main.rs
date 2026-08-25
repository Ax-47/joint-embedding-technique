use data_process::read_byte::DataSet;
use ndarray::Array1;
use neural_networks::activation_functions::{Relu, relu};
use std::io::Result;
use utils::{
    functors::CollectionFunctor,
    morphism::{self, Morphism},
};

fn main() -> Result<()> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let l1 = neural_networks::layers::linear::new_linear_layer(784, 64);
    let l2 = neural_networks::layers::linear::new_linear_layer(64, 64);
    let l3 = neural_networks::layers::linear::new_linear_layer(64, 10);
    let net = l1
        .compose::<_, Array1<f64>>(CollectionFunctor::new(Relu))
        .compose(l2)
        .compose::<_, Array1<f64>>(CollectionFunctor::new(Relu))
        .compose(l3);
    for epoch in 0..10 {
        for (batch_idx, batch) in dataset.batch_view_iter(1, 60_000).enumerate() {
            let labels = batch.label_one_hot();
            let al = net.apply(batch.images_vecf64().into_flat()).unwrap();
            let cost = &al - &labels.into_flat();
            let cost = cost.pow2().sum();

            println!("epoch:              {}/10", epoch + 1);
            // println!("batch:              {}", batch_idx);
            // println!("train times:        {}", trained_time);
            println!("C                 = {}", cost);
            // println!("loss              = {}", loss);
            // println!("Learning Rate     = {:?}", nn.get_learning_rate());
            // trained_time += 1;
        }
    }
    Ok(())
}
