use joint_embedding_core::{siamese_clr::train_siamese_clr, siamese_net::train_siamese_net};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // train_mlp()?;
    // train_siamese_net()?;
    train_siamese_clr()?;

    Ok(())
}
