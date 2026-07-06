// Copyright 2026 The Binius Developers

use binius_core::{
	ShiftVariant,
	constraint_system::{
		AndConstraint, ConstraintSystem, MulConstraint, Operand, ShiftedValueIndex,
	},
};
use binius_verifier::{
	config::WORD_SIZE_BITS,
	protocols::shift::{BITAND_ARITY, INTMUL_ARITY, SHIFT_VARIANT_COUNT},
};

/// One (constraint, word) reference of a wiring matrix.
///
/// `word_index` is segment-relative — it indexes into the segment's slice of the witness `words`.
/// `constraint_index` indexes into the operation's constraint list, and thus into the operation's
/// `r_x_prime_tensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiringEntry {
	pub constraint_index: u32,
	pub word_index: u32,
}

/// The sparse matrix for a fixed (operation, operand, shift variant, shift amount).
///
/// This is the wiring matrix `M` from the paper, restricted to a single 2D slice: it lists every
/// (constraint, word) reference in which a word of this segment appears as the given operand of the
/// given operation, shifted by the given variant and amount.
pub type WiringMatrix = Vec<WiringEntry>;

/// The wiring matrices of one value-vector segment, transposed from the per-word [`KeySegment`]
/// layout into a per-(operation, operand, shift variant, shift amount) layout.
///
/// Where a [`KeySegment`] is indexed by word, a `WiringInfo` is indexed by (operation, operand,
/// shift variant, shift amount) and stores, for each such tuple, the sparse matrix of
/// constraint/word references. This is the data structure the alternate
/// [`build_g_parts_wiring`](super::phase_1::build_g_parts_wiring) algorithm iterates over.
///
/// Operand ordering matches the [`KeySegment`] convention: bitand `[a, b, c]`, intmul
/// `[a, b, lo, hi]` (so operand 2 = `lo`, operand 3 = `hi`), keeping `lambda_powers[operand]`
/// aligned. The `*32` variants (`Sll32`/`Srl32`/`Sra32`/`Rotr32`) only ever use amounts `0..32`, so
/// half their matrices stay empty.
///
/// [`KeySegment`]: super::key_collection::KeySegment
pub struct WiringInfo {
	/// The number of witness words this segment covers.
	pub n_words: usize,
	/// Indexed `[operand][shift_variant][shift_amount]`.
	pub bitand: [[[WiringMatrix; WORD_SIZE_BITS]; SHIFT_VARIANT_COUNT]; BITAND_ARITY],
	/// Indexed `[operand][shift_variant][shift_amount]`.
	pub intmul: [[[WiringMatrix; WORD_SIZE_BITS]; SHIFT_VARIANT_COUNT]; INTMUL_ARITY],
}

impl WiringInfo {
	/// An empty `WiringInfo` covering `n_words` words, with every matrix empty.
	fn empty(n_words: usize) -> Self {
		Self {
			n_words,
			bitand: std::array::from_fn(|_| {
				std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()))
			}),
			intmul: std::array::from_fn(|_| {
				std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()))
			}),
		}
	}
}

/// The wiring matrices of a constraint system, split by value-vector segment.
///
/// Mirrors the [`KeyCollection`](super::key_collection::KeyCollection) `{ public, hidden }` split:
/// one [`WiringInfo`] for the public words (value-vector indices `[0, n_public_words)`) and one for
/// the hidden words (indices `[n_public_words, committed_total_len)`). Word indices within each
/// segment are segment-relative.
pub struct WiringCollection {
	pub public: WiringInfo,
	pub hidden: WiringInfo,
}

/// Maps a [`ShiftVariant`] to its index in `0..SHIFT_VARIANT_COUNT`, matching the encoding used by
/// `KeyCollection`.
#[inline]
const fn shift_variant_index(shift_variant: ShiftVariant) -> usize {
	match shift_variant {
		ShiftVariant::Sll => 0,
		ShiftVariant::Slr => 1,
		ShiftVariant::Sar => 2,
		ShiftVariant::Rotr => 3,
		ShiftVariant::Sll32 => 4,
		ShiftVariant::Srl32 => 5,
		ShiftVariant::Sra32 => 6,
		ShiftVariant::Rotr32 => 7,
	}
}

/// Routes every (constraint, word) reference of one operand into its segment's `[variant][amount]`
/// matrix, choosing the public or hidden segment by the absolute word index.
fn route_operand<'a>(
	public_matrices: &mut [[WiringMatrix; WORD_SIZE_BITS]; SHIFT_VARIANT_COUNT],
	hidden_matrices: &mut [[WiringMatrix; WORD_SIZE_BITS]; SHIFT_VARIANT_COUNT],
	n_public_words: u32,
	operands: impl Iterator<Item = &'a Operand>,
) {
	for (constraint_index, operand) in operands.enumerate() {
		for &ShiftedValueIndex {
			value_index,
			shift_variant,
			amount,
		} in operand
		{
			let variant = shift_variant_index(shift_variant);
			// Segments partition the words at `n_public_words`; make the index segment-relative.
			let (matrices, word_index) = if value_index.0 < n_public_words {
				(&mut *public_matrices, value_index.0)
			} else {
				(&mut *hidden_matrices, value_index.0 - n_public_words)
			};
			matrices[variant][amount as usize].push(WiringEntry {
				constraint_index: constraint_index as u32,
				word_index,
			});
		}
	}
}

/// Constructs the public and hidden [`WiringInfo`]s from a constraint system.
///
/// This iterates the constraints once, bucketing each operand's shifted value references by
/// (segment, operation, operand, shift variant, shift amount) — the same enumeration
/// `build_key_collection` performs, transposed.
pub fn build_wiring_info(cs: &ConstraintSystem) -> WiringCollection {
	let n_public_words = cs.value_vec_layout.n_public_words();
	let mut public = WiringInfo::empty(n_public_words);
	let mut hidden = WiringInfo::empty(cs.value_vec_layout.n_hidden_words());

	let bitand_operand_getters: [fn(&AndConstraint) -> &Operand; BITAND_ARITY] =
		[|c| &c.a, |c| &c.b, |c| &c.c];
	let intmul_operand_getters: [fn(&MulConstraint) -> &Operand; INTMUL_ARITY] =
		[|c| &c.a, |c| &c.b, |c| &c.lo, |c| &c.hi];

	for (operand_index, get_operand) in bitand_operand_getters.iter().enumerate() {
		route_operand(
			&mut public.bitand[operand_index],
			&mut hidden.bitand[operand_index],
			n_public_words as u32,
			cs.and_constraints.iter().map(get_operand),
		);
	}
	for (operand_index, get_operand) in intmul_operand_getters.iter().enumerate() {
		route_operand(
			&mut public.intmul[operand_index],
			&mut hidden.intmul[operand_index],
			n_public_words as u32,
			cs.mul_constraints.iter().map(get_operand),
		);
	}

	WiringCollection { public, hidden }
}
