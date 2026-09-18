use joint_embedding_core::{
    barlow::train_barlow, mlp::train_mlp, siamese_clr::train_siamese_clr,
    siamese_net::train_siamese_net,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    train_mlp()?;
    // train_siamese_net()?;
    // train_siamese_clr()?;
    // train_barlow()?;

    Ok(())
}
