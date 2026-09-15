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
    Backend(String),
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

            CategoryError::Backend(msg) => write!(f, "backend error: {msg}"),
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
impl From<candle_core::Error> for CategoryError {
    fn from(e: candle_core::Error) -> Self {
        CategoryError::Backend(e.to_string()) // ปรับตาม variant ที่คุณมี
    }
}
pub type CategoryResult<Output> = Result<Output, CategoryError>;
