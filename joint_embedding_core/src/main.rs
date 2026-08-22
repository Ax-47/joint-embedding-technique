use data_process::read_byte::DataSet;
use ndarray::Array1;
use std::io::Result;
use utils::morphism::{self, Morphism};

fn main() -> Result<()> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    let l1 = neural_networks::layers::LinearLayer::new(
        784,
        64,
        utils::activation_function::ActivationFunctionPair::relu(),
    );

    let l2 = neural_networks::layers::LinearLayer::new(
        64,
        64,
        utils::activation_function::ActivationFunctionPair::relu(),
    );

    let l3 = neural_networks::layers::LinearLayer::new(
        64,
        10,
        utils::activation_function::ActivationFunctionPair::relu(),
    );
    let l1_l2 = morphism::Compose::<_, _, Array1<f64>>::new(l1, l2);
    let l1_l2_l3 = morphism::Compose::<_, _, Array1<f64>>::new(l1_l2, l3);
    for epoch in 0..10 {
        for (batch_idx, batch) in dataset.batch_view_iter(1, 60_000).enumerate() {
            let labels = batch.label_one_hot();
            let al = l1_l2_l3.apply(batch.images_vecf64().into_flat()).unwrap();
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
