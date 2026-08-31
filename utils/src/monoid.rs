pub trait Monoid: Sized {
    fn empty() -> Self;
    fn combine(&self, other: &Self) -> Self;
}
