// Copyright 2025 Irreducible Inc.
use cranelift_entity::PrimaryMap;

/// A designator of a path within a circuit.
///
/// Compact, only 32-bit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathSpec(u32);
cranelift_entity::entity_impl!(PathSpec);

struct Node {
	/// Offset of the name within the tree's name buffer.
	start: u32,
	/// Length of the name in bytes.
	len: u32,
	parent: PathSpec,
}

/// A tree that holds paths within a circuit.
pub struct PathSpecTree {
	root: PathSpec,
	nodes: PrimaryMap<PathSpec, Node>,
	/// Every node name, concatenated in the order the nodes were created.
	///
	/// Names are only ever appended.
	/// A node's range therefore stays valid for the life of the tree.
	names: String,
}

impl PathSpecTree {
	/// Creates a new empty tree.
	pub fn new() -> Self {
		let mut nodes = PrimaryMap::new();
		let root = nodes.push(Node {
			start: 0,
			len: 0,
			parent: PathSpec(0),
		});
		Self {
			root,
			nodes,
			names: String::new(),
		}
	}

	/// Extend the tree with a new branch that stems from the given `parent` and has a certain
	/// `name`.
	pub fn extend(&mut self, parent: PathSpec, name: &str) -> PathSpec {
		let start = self.names.len() as u32;
		self.names.push_str(name);
		self.nodes.push(Node {
			start,
			len: name.len() as u32,
			parent,
		})
	}

	/// Writes a string representation of the given path spec into a given string buffer.
	///
	/// Every component is written with a leading `'.'`, including the first.
	/// So a path one level below the root renders as `.name`, not `name`.
	///
	/// The buffer is appended to rather than reset.
	/// A caller that reuses a buffer therefore has to clear it first.
	pub fn stringify(&self, ls: PathSpec, out: &mut String) {
		fn stringify_rec(tree: &PathSpecTree, ls: PathSpec, out: &mut String) {
			if ls == tree.root {
				return;
			}
			let node = &tree.nodes[ls];
			stringify_rec(tree, node.parent, out);
			out.push('.');
			out.push_str(&tree.names[node.start as usize..(node.start + node.len) as usize]);
		}
		stringify_rec(self, ls, out);
	}

	/// Returns the parent of the given path or null if `root` was supplied.
	pub fn parent(&self, path: PathSpec) -> Option<PathSpec> {
		if path == self.root {
			return None;
		}
		Some(self.nodes[path].parent)
	}

	/// Returns the root of the tree.
	pub const fn root(&self) -> PathSpec {
		self.root
	}
}

impl Default for PathSpecTree {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn every_node_renders_its_own_name_after_later_nodes_are_added() {
		let mut tree = PathSpecTree::new();
		let root = tree.root();

		// Two branches under the root, each extended after the other was created.
		let a = tree.extend(root, "alpha");
		let a_child = tree.extend(a, "one");
		let b = tree.extend(root, "b");
		let b_child = tree.extend(b, "considerably_longer");

		let render = |path| {
			let mut out = String::new();
			tree.stringify(path, &mut out);
			out
		};

		assert_eq!(render(root), "");
		assert_eq!(render(a), ".alpha");
		assert_eq!(render(a_child), ".alpha.one");
		assert_eq!(render(b), ".b");
		assert_eq!(render(b_child), ".b.considerably_longer");
	}

	#[test]
	fn an_empty_name_renders_as_a_bare_separator() {
		let mut tree = PathSpecTree::new();
		let root = tree.root();
		let empty = tree.extend(root, "");
		let below = tree.extend(empty, "leaf");

		let mut out = String::new();
		tree.stringify(below, &mut out);
		assert_eq!(out, "..leaf");
	}

	#[test]
	fn the_root_has_no_parent() {
		let mut tree = PathSpecTree::new();
		let root = tree.root();
		let child = tree.extend(root, "child");

		assert_eq!(tree.parent(root), None);
		assert_eq!(tree.parent(child), Some(root));
	}
}
