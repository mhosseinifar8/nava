//! Nava Core Rust - Ultra-fast Persian text normalization and streaming chunking.

pub mod chunker;
pub mod normalizer;

pub use chunker::{chunk_by_token_budget, split_into_streaming_clauses};
pub use normalizer::{clean_for_matcha, normalize_persian};

