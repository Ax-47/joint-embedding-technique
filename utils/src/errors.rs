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
            Self::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "shape mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::NumericalError(msg) => write!(f, "numerical error: {msg}"),
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl Error for CategoryError {}

impl From<ShapeError> for CategoryError {
    fn from(error: ShapeError) -> Self {
        Self::InvalidInput(format!("ndarray shape error: {error}"))
    }
}

impl From<candle_core::Error> for CategoryError {
    fn from(error: candle_core::Error) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<std::io::Error> for CategoryError {
    fn from(error: std::io::Error) -> Self {
        Self::Backend(format!("io error: {error}"))
    }
}

pub type CategoryResult<T> = Result<T, CategoryError>;
