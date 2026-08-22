#[derive(Debug, Clone)]
pub enum CategoryError {}
pub type CategoryResult<Output> = Result<Output, CategoryError>;
