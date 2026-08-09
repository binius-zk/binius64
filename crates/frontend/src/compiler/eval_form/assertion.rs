// Copyright 2026 The Binius Developers
//! What the single-instance and batched execution contexts share about assertion failures.

use crate::compiler::pathspec::{PathSpec, PathSpecTree};

/// The cap on how many assertion failures an execution context retains.
///
/// Failures past the cap are counted but not stored.
pub const MAX_ASSERTION_FAILURES: usize = 100;

/// Renders the circuit path an assertion was raised under.
///
/// Empty when no tree is available to resolve the path, or when the assertion sits at the root.
/// The path is kept apart from the failure detail so a caller can group failures by subcircuit.
pub fn render_path(path_spec_tree: Option<&PathSpecTree>, path_spec: PathSpec) -> String {
	let Some(tree) = path_spec_tree else {
		return String::new();
	};

	let mut path = String::new();
	tree.stringify(path_spec, &mut path);
	path
}
