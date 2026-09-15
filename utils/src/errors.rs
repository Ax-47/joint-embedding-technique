use std::{error::Error, fmt};

use ndarray::ShapeError;

#[derive(Debug, Clone)]
pub enum CategoryError {
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    InvalidInput(String),
    NumericalError(String),
}

impl fmt::Display for CategoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CategoryError::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "shape mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
            CategoryError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            CategoryError::NumericalError(msg) => write!(f, "numerical error: {msg}"),
        }
    }
}

impl Error for CategoryError {}
impl From<ShapeError> for CategoryError {
    fn from(e: ShapeError) -> Self {
        CategoryError::ShapeMismatch {
            expected: vec![],
            found: vec![],
        }
    }
}
pub type CategoryResult<Output> = Result<Output, CategoryError>;
