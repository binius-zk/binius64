// Copyright 2026 The Binius Developers
//! What the single-instance and batched execution contexts share about assertion failures.

use crate::compiler::pathspec::{PathSpec, PathSpecTree};

/// The cap on how many assertion failures an execution context retains.
///
/// Failures past the cap are counted but not stored.
pub const MAX_ASSERTION_FAILURES: usize = 100;

/// Renders a failure message, prefixed by the path the assertion was raised under.
///
/// The prefix is dropped when no tree is available to resolve the path, or when it renders empty.
pub fn symbolicate(
	path_spec_tree: Option<&PathSpecTree>,
	path_spec: PathSpec,
	message: String,
) -> String {
	let Some(tree) = path_spec_tree else {
		return message;
	};

	let mut path = String::new();
	tree.stringify(path_spec, &mut path);
	if path.is_empty() {
		message
	} else {
		format!("{path}: {message}")
	}
}
