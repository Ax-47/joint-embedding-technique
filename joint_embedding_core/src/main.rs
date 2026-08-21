use data_process::read_byte::DataSet;
use std::io::Result;

fn main() -> Result<()> {
    let dataset = DataSet::new(
        "MNIST/train-images-idx3-ubyte",
        "MNIST/train-labels-idx1-ubyte",
    )?;
    Ok(())
}
