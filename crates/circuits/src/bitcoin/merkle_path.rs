// Copyright 2025 Irreducible Inc.
//! The merkle proof for a transaction in a Bitcoin block.

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::double_sha256::double_sha256;

/// Which side of a double-SHA256 pair a merkle sibling sits on.
#[derive(Debug, Copy, Clone)]
pub enum SiblingSide {
	Left,
	Right,
}

impl SiblingSide {
	/// The wire value the side is passed to [`merkle_path`] as.
	///
	/// The side wire is read as a `select` condition, which tests the most significant bit, so
	/// the two sides must be the all-zero and all-one words rather than `0` and `1`.
	pub const fn to_word(self) -> Word {
		match self {
			SiblingSide::Left => Word::ZERO,
			SiblingSide::Right => Word::ALL_ONE,
		}
	}
}

/// Returns the merkle root obtained by folding `leaf` with `siblings` under double-SHA256.
///
/// Every wire is Bitcoin little-endian packed (8 bytes per wire).
///
/// Each sibling carries a wire saying whether it is a left or a right sibling, in the encoding
/// [`SiblingSide::to_word`] produces. `siblings.len()` is the maximal path length; only the first
/// `length` siblings are folded in, and the remaining levels pass the running digest through
/// unchanged.
pub fn merkle_path(
	builder: &CircuitBuilder,
	mut leaf: [Wire; 4],
	siblings: &[([Wire; 4], Wire)],
	length: Wire,
) -> [Wire; 4] {
	for (i, (sibling, is_right)) in siblings.iter().enumerate() {
		// The pair is hashed in path order: a right sibling is appended after the running
		// digest, a left one is prepended.
		let message: Vec<Wire> = (0..4)
			.map(|j| builder.select(*is_right, leaf[j], sibling[j]))
			.chain((0..4).map(|j| builder.select(*is_right, sibling[j], leaf[j])))
			.collect();
		let digest = double_sha256(builder, &message);

		// Levels at or past `length` are padding: keep the digest reached so far.
		let within_length = builder.icmp_ult(builder.add_constant_64(i as u64), length);
		leaf = std::array::from_fn(|j| builder.select(within_length, digest[j], leaf[j]));
	}
	leaf
}

#[cfg(test)]
mod tests {
	use std::array;

	use hex_literal::hex;

	use super::*;

	/// Builds a circuit asserting `merkle_path(leaf, siblings, length) == root`, then runs it.
	fn check_merkle_path(
		max_path_len: usize,
		leaf_value: [u8; 32],
		siblings_value: &[([u8; 32], SiblingSide)],
		length_value: u64,
		root_value: [u8; 32],
	) -> anyhow::Result<()> {
		let builder = CircuitBuilder::new();
		let leaf: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		let siblings: Vec<([Wire; 4], Wire)> = std::iter::repeat_with(|| {
			(array::from_fn(|_| builder.add_witness()), builder.add_witness())
		})
		.take(max_path_len)
		.collect();
		let root: [Wire; 4] = array::from_fn(|_| builder.add_witness());
		let length = builder.add_witness();
		builder.assert_eq_v("root", merkle_path(&builder, leaf, &siblings, length), root);
		let circuit = builder.build();

		let mut filler = circuit.new_witness_filler();
		filler.pack_bytes_le(&leaf, &leaf_value);
		for ((sibling, is_right), (value, side)) in siblings.iter().zip(siblings_value) {
			filler.pack_bytes_le(sibling, value);
			filler[*is_right] = side.to_word();
		}
		filler.pack_bytes_le(&root, &root_value);
		filler[length] = Word(length_value);
		circuit.populate_wire_witness(&mut filler)?;

		let constraint_system = circuit.constraint_system();
		constraint_system.verify(&filler.into_value_vec())?;
		Ok(())
	}

	const LEAF: [u8; 32] = hex!("a2b6b171aae6007508e5c8fabec6b662bad3e4594e09405cac7b249e5f1e5155");
	const ROOT: [u8; 32] = hex!("5802c63ef536216cf01a0dd0b32c01f5e31536aa773eb6e1d46fd42f66516eba");
	const SIBLING_0: [u8; 32] =
		hex!("1346be1a16a09b5fcc5bca52d39c2529396f0fa6a654f3978807ff79eaf91d66");
	const SIBLING_1: [u8; 32] =
		hex!("557cc3606e7197ff5a7b6cda46e409445b1ab58d8d4ebf1bc3d95764c32ad877");

	#[test]
	fn test_valid() {
		let siblings = [
			(SIBLING_0, SiblingSide::Right),
			(SIBLING_1, SiblingSide::Left),
		];
		check_merkle_path(2, LEAF, &siblings, 2, ROOT).unwrap();
	}

	#[test]
	fn test_invalid_side() {
		// Flipping the second sibling's side reverses the order the pair is hashed in.
		let siblings = [
			(SIBLING_0, SiblingSide::Right),
			(SIBLING_1, SiblingSide::Right),
		];
		check_merkle_path(2, LEAF, &siblings, 2, ROOT).unwrap_err();
	}

	#[test]
	fn test_invalid_path() {
		let wrong = hex!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
		let siblings = [(SIBLING_0, SiblingSide::Right), (wrong, SiblingSide::Left)];
		check_merkle_path(2, LEAF, &siblings, 2, ROOT).unwrap_err();
	}

	/// A 12-level path in a circuit sized for 30, so the 18 padding levels must pass through.
	#[test]
	fn test_valid_long() {
		let leaf = hex!("6f2f044a225e8b293c6e54cf2771bf4d17ba8904b1f61cf9c392965dcbda0b83");
		let root = hex!("fc01df2139954b36cebc3fa6fbf6a7160a67d34b67e5c4aa2a7ce46f5bb42a83");
		let siblings = [
			(
				hex!("783089645b0bc42d44e9d6a7ea62adf7a8a2adc6b7f0173d663369217b771b86"),
				SiblingSide::Right,
			),
			(
				hex!("1eeeeb0cac1753a10ade3b34bd5bf0e005cdec82545abdafa38685c45e5f8ce5"),
				SiblingSide::Right,
			),
			(
				hex!("e0d7426d603f1a817938cf366c8933d32185625fc821e3b1e964cb5f8e421501"),
				SiblingSide::Right,
			),
			(
				hex!("7af6e333025422cf892198d216f146d70efe64119071ce0ee96fd195640230df"),
				SiblingSide::Left,
			),
			(
				hex!("d848bf00d7563a26c9a43ad8cc2fa558f6a299629be20a078a6b197dcf15fc31"),
				SiblingSide::Right,
			),
			(
				hex!("b643abf3df379ac748494a5eb3025299265fff543571f8b71935e533f672c9e8"),
				SiblingSide::Right,
			),
			(
				hex!("b07d3ebc129da3ae9d1b9daee64daf74f8504ca5f9194cd006edee48b1bf4d00"),
				SiblingSide::Right,
			),
			(
				hex!("4cd4173f585e793e48aa479269f38cd986b600c494135e9de33118a8e4ac03ed"),
				SiblingSide::Right,
			),
			(
				hex!("4b5e59b8d22762cfc2906fa597b29c7eab7cd52d4b0cea9269e84e2aebce4101"),
				SiblingSide::Right,
			),
			(
				hex!("2321cd016cb8f1a29f1bad981418bed2776bf61b1a729ca86a54f14790ce822b"),
				SiblingSide::Right,
			),
			(
				hex!("fecdc8a219a271a9a969fdebf38068ffeaf25b7af353ee99e759eb0d05604218"),
				SiblingSide::Right,
			),
			(
				hex!("1736c19cc6de7296453811916ddedba46c9bbd61a3450ad3dfb8bddb698b6ad0"),
				SiblingSide::Right,
			),
		];
		check_merkle_path(30, leaf, &siblings, 12, root).unwrap();
	}
}
