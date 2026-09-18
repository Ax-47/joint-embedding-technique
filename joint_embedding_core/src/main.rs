use crate::{mlp::train_mlp, siamese_net::train_siamese_net};

mod mlp;
mod siamese_net;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // train_mlp()?;
    train_siamese_net()?;
    Ok(())
}
