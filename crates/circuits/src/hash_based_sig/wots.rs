// Copyright 2026 The Binius Developers
// Copyright (c) 2026 leanEthereum
//! WOTS (Winternitz one-time signature) with target-sum encoding.
//!
//! There are no checksum chains. Instead the signer grinds the signature randomness until the
//! encoding's digits sum to [`TARGET_SUM`], and a verifier that checks the sum knows no digit can
//! have been lowered without another being raised.

use std::iter;

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Hint, Wire};
use rand::CryptoRng;

use super::{
	CHAIN_LENGTH, DIGEST_LEN, DIGEST_WIRES, Digest, MESSAGE_LEN, MESSAGE_WIRES, Message,
	NUM_CHAIN_HASHES, PUBLIC_PARAM_LEN, PUBLIC_PARAM_WIRES, PublicParam, RANDOMNESS_LEN,
	RANDOMNESS_WIRES, Randomness, TARGET_SUM, V, W,
	hashing::{
		TWEAK_TYPE_CHAIN, TWEAK_TYPE_ENCODING, TWEAK_TYPE_WOTS_PK, circuit_tweak_hash,
		circuit_tweak_hash_2x, tweak_hash,
	},
};
use crate::multiplexer::multi_wire_multiplex;

/// Digits carried by each of the digest's two 64-bit words.
const DIGITS_PER_WORD: usize = V / 2;

/// The encoding hashes `message | randomness`, which is one BLAKE3 block with room to spare now
/// that the domain rides in the key rather than the payload.
const ENCODING_PAYLOAD_LEN: usize = MESSAGE_LEN + RANDOMNESS_LEN;

const _: () = assert!(ENCODING_PAYLOAD_LEN <= 64);
const _: () = assert!(2 * DIGITS_PER_WORD == V);

/// The target-sum encoding.
///
/// `D` is the encoding hash of `message | randomness`, truncated to 16 bytes. Each of its two
/// little-endian 64-bit words holds 21 digits of [`W`] bits: digit `i < 21` at bits `3i` of word
/// 0, digit `i >= 21` at bits `3(i - 21)` of word 1.
///
/// The encoding is valid exactly when the leftover top bit of *each* word (bits 63 and 127) is
/// zero and the digits sum to [`TARGET_SUM`]. Grinding the top bits to zero makes each word
/// exactly `sum(e_i * 2^{3i})` over its 21 digits, so both words decompose into digits with no
/// slack term.
///
/// Returns `None` when the randomness does not produce a valid encoding, which is the signal the
/// grinding loop retries on.
pub fn wots_encode(
	message: &Message,
	epoch: u32,
	public_param: &PublicParam,
	randomness: &Randomness,
) -> Option<[u8; V]> {
	let mut data = [0u8; ENCODING_PAYLOAD_LEN];
	data[..MESSAGE_LEN].copy_from_slice(message);
	data[MESSAGE_LEN..][..RANDOMNESS_LEN].copy_from_slice(randomness);
	let digest = tweak_hash(public_param, TWEAK_TYPE_ENCODING, 0, epoch, &data);

	if digest[7] >> 7 != 0 || digest[DIGEST_LEN - 1] >> 7 != 0 {
		return None; // the leftover top bit of each 64-bit word must be zero
	}
	let bit = |j: usize| (digest[j / 8] >> (j % 8)) & 1;
	let pos = |i: usize| {
		if i < DIGITS_PER_WORD {
			W * i
		} else {
			64 + W * (i - DIGITS_PER_WORD)
		}
	};
	let encoding: [u8; V] =
		std::array::from_fn(|i| (0..W).fold(0, |acc, k| acc | (bit(pos(i) + k) << k)));
	(encoding.iter().map(|&x| x as usize).sum::<usize>() == TARGET_SUM).then_some(encoding)
}

/// Draws randomness until it encodes validly.
///
/// The encoding is valid when both leftover bits are zero and the digits hit the target sum, so
/// grinding takes fewer than `2^15` attempts on average.
pub fn find_randomness_for_wots_encoding(
	message: &Message,
	epoch: u32,
	public_param: &PublicParam,
	rng: &mut impl CryptoRng,
) -> (Randomness, [u8; V]) {
	loop {
		let mut randomness = [0u8; RANDOMNESS_LEN];
		rng.fill_bytes(&mut randomness);
		if let Some(encoding) = wots_encode(message, epoch, public_param, &randomness) {
			return (randomness, encoding);
		}
	}
}

/// One chain step.
///
/// The position `chain_index * CHAIN_LENGTH + step` identifies the edge from chain value `step` to
/// `step + 1`, so no two edges anywhere in the instance share a tweak.
pub fn chain_step(
	public_param: &PublicParam,
	epoch: u32,
	chain_index: usize,
	step: usize,
	x: &Digest,
) -> Digest {
	let position = (chain_index * CHAIN_LENGTH + step) as u32;
	tweak_hash(public_param, TWEAK_TYPE_CHAIN, position, epoch, x)
}

/// Walks chain `chain_index` for `n` steps starting at chain value `start_step`.
pub fn iterate_hash(
	a: &Digest,
	n: usize,
	public_param: &PublicParam,
	epoch: u32,
	chain_index: usize,
	start_step: usize,
) -> Digest {
	(0..n).fold(*a, |acc, j| chain_step(public_param, epoch, chain_index, start_step + j, &acc))
}

/// Walks every chain from its tip to the public-key end.
pub fn recover_public_key(
	chain_tips: &[Digest; V],
	encoding: &[u8; V],
	epoch: u32,
	public_param: &PublicParam,
) -> [Digest; V] {
	std::array::from_fn(|i| {
		let digit = encoding[i] as usize;
		iterate_hash(&chain_tips[i], CHAIN_LENGTH - 1 - digit, public_param, epoch, i, digit)
	})
}

/// The Merkle leaf: the hash over the public parameter and the [`V`] concatenated chain ends.
pub fn wots_public_key_hash(
	public_param: &PublicParam,
	epoch: u32,
	chain_ends: &[Digest; V],
) -> Digest {
	let mut data = [0u8; V * DIGEST_LEN];
	for (chunk, end) in iter::zip(data.chunks_exact_mut(DIGEST_LEN), chain_ends) {
		chunk.copy_from_slice(end);
	}
	tweak_hash(public_param, TWEAK_TYPE_WOTS_PK, 0, epoch, &data)
}

/// In-circuit form of [`wots_encode`], returning the digits and constraining them to be a valid
/// encoding.
///
/// Both validity conditions are asserted rather than returned: an encoding whose leftover bits are
/// set, or whose digits miss the target sum, has no satisfying witness.
///
/// # Returns
///
/// The [`V`] digits, each a wire holding a value below [`CHAIN_LENGTH`].
pub fn circuit_wots_encode(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	message: &[Wire; MESSAGE_WIRES],
	randomness: &[Wire; RANDOMNESS_WIRES],
) -> [Wire; V] {
	let zero = builder.add_constant_64(0);

	// `message | randomness`, which fills its wires exactly.
	let mut payload = Vec::with_capacity(ENCODING_PAYLOAD_LEN / 8);
	payload.extend_from_slice(message);
	payload.extend_from_slice(randomness);

	let digest =
		circuit_tweak_hash(builder, public_param, TWEAK_TYPE_ENCODING, zero, epoch, &payload);

	// With `V * W = 126` of the digest's 128 bits spent on digits, the two leftover top bits are
	// what would otherwise let a word carry a slack term the digits do not account for.
	for (k, &word) in digest.iter().enumerate() {
		builder.assert_zero(format!("encoding_leftover_bit[{k}]"), builder.shr(word, 63));
	}

	// A digit is W bits from the middle of its word: lift them to the top of the word, then drop
	// them back to the bottom, so nothing above or below them survives.
	let digits: [Wire; V] = std::array::from_fn(|i| {
		let word = digest[i / DIGITS_PER_WORD];
		let shift = (W * (i % DIGITS_PER_WORD)) as u32;
		builder.shr(builder.shl(word, u64::BITS - shift - W as u32), u64::BITS - W as u32)
	});

	// The digits are each below CHAIN_LENGTH by construction, so the sum cannot overflow.
	let sum = digits
		.iter()
		.fold(zero, |acc, &digit| builder.iadd(acc, digit).0);
	builder.assert_eq("encoding_target_sum", sum, builder.add_constant_64(TARGET_SUM as u64));

	digits
}

/// Words the hint emits per chain hash: the input digest, then the chain and the step.
const HINT_WORDS_PER_HASH: usize = DIGEST_WIRES + 2;

/// Words the hint reads.
const HINT_INPUTS: usize = PUBLIC_PARAM_WIRES + 1 + V + V * DIGEST_WIRES;

/// Words the hint writes: one entry per chain hash, then the [`V`] chain ends.
const HINT_OUTPUTS: usize = NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH + V * DIGEST_WIRES;

/// Computes the chain hashes a verifier actually walks, and where each one sits.
///
/// The verifier's work is the concatenation of the chain tails: chain `i` contributes its steps
/// from `digit_i` up to `CHAIN_LENGTH - 2`, and the tails run in chain order. How long each tail
/// is depends on the digits, so the list cannot be laid out at circuit construction time — it is
/// hinted here and pinned by the constraints in [`circuit_recover_public_key`].
struct ChainHashesHint;

impl Hint for ChainHashesHint {
	const NAME: &'static str = "binius.xmss_wots_chain_hashes";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(HINT_INPUTS, HINT_OUTPUTS)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let public_param = bytes_from_words::<PUBLIC_PARAM_LEN>(&inputs[..PUBLIC_PARAM_WIRES]);
		let epoch = inputs[PUBLIC_PARAM_WIRES].as_u64() as u32;
		let digits = &inputs[PUBLIC_PARAM_WIRES + 1..][..V];
		let tips = &inputs[PUBLIC_PARAM_WIRES + 1 + V..];

		let (hashes, ends) = outputs.split_at_mut(NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH);
		hashes.fill(Word::ZERO);

		let mut written = 0;
		for i in 0..V {
			let digit = digits[i].as_u64() as usize;
			let mut current =
				bytes_from_words::<DIGEST_LEN>(&tips[i * DIGEST_WIRES..][..DIGEST_WIRES]);

			// A chain walks from its digit to the last step. A digit of `CHAIN_LENGTH - 1` walks
			// nothing, and a digit past that (an unsatisfiable witness) walks nothing either.
			for step in digit..CHAIN_LENGTH - 1 {
				// Digits that miss the target sum overrun the list. The encoding constraints
				// reject them, so stopping short here only has to avoid a panic.
				if written == NUM_CHAIN_HASHES {
					break;
				}
				let slot = &mut hashes[written * HINT_WORDS_PER_HASH..][..HINT_WORDS_PER_HASH];
				bytes_to_words(&current, &mut slot[..DIGEST_WIRES]);
				slot[DIGEST_WIRES] = Word::from_u64(i as u64);
				slot[DIGEST_WIRES + 1] = Word::from_u64(step as u64);

				current = chain_step(&public_param, epoch, i, step, &current);
				written += 1;
			}
			bytes_to_words(&current, &mut ends[i * DIGEST_WIRES..][..DIGEST_WIRES]);
		}
	}
}

/// Little-endian bytes from 64-bit words.
fn bytes_from_words<const N: usize>(words: &[Word]) -> [u8; N] {
	let mut bytes = [0u8; N];
	for (chunk, word) in iter::zip(bytes.chunks_exact_mut(8), words) {
		chunk.copy_from_slice(&word.as_u64().to_le_bytes());
	}
	bytes
}

/// The inverse of [`bytes_from_words`].
fn bytes_to_words(bytes: &[u8], words: &mut [Word]) {
	for (word, chunk) in iter::zip(words, bytes.chunks_exact(8)) {
		*word = Word::from_u64(u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
	}
}

/// In-circuit form of [`recover_public_key`], spending only the hashes a verifier walks.
///
/// A chain's tail is `CHAIN_LENGTH - 1 - digit` hashes, and the target sum fixes the total across
/// all chains at [`NUM_CHAIN_HASHES`] however the digits fall. So rather than give every chain
/// room for its longest possible tail — `V * (CHAIN_LENGTH - 1)` hashes, two thirds of them
/// discarded — the tails are concatenated into one list of exactly that total, hinted, and pinned
/// by constraints. Each entry carries its input digest, its chain and its step; its output is the
/// hash, not a hinted value, so nothing has to check it.
///
/// # What pins the list
///
/// - `chain` is non-decreasing, so each chain's entries are one contiguous run.
/// - Within a run, the step advances by one and each input is the previous output.
/// - A run's first entry takes its chain's signature tip as input and starts at `digit`; its last
///   ends at `CHAIN_LENGTH - 2` and its output is that chain's end.
/// - A chain whose digit is `CHAIN_LENGTH - 1` owes no hashes; its end is its tip.
///
/// A run's length therefore has to be exactly `CHAIN_LENGTH - 1 - digit`, and the list is exactly
/// [`NUM_CHAIN_HASHES`] long, which the target sum makes the sum of those lengths. **No chain that
/// owes hashes can be missing a run**: if one were, the entries would not add up.
///
/// Every entry looks its chain's tip, digit and end up in one [`V`]-entry table, indexed by the
/// entry's own `chain` — which is where the variable layout is paid for, in committed words rather
/// than in hashes.
pub fn circuit_recover_public_key(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	chain_tips: &[[Wire; DIGEST_WIRES]; V],
	digits: &[Wire; V],
) -> [[Wire; DIGEST_WIRES]; V] {
	let mut hint_inputs = Vec::with_capacity(HINT_INPUTS);
	hint_inputs.extend_from_slice(public_param);
	hint_inputs.push(epoch);
	hint_inputs.extend_from_slice(digits);
	hint_inputs.extend(chain_tips.iter().flatten().copied());
	let hinted = builder.call_hint(ChainHashesHint, &[], &hint_inputs);

	let input_of = |k: usize| -> [Wire; DIGEST_WIRES] {
		std::array::from_fn(|w| hinted[k * HINT_WORDS_PER_HASH + w])
	};
	let chain_of = |k: usize| hinted[k * HINT_WORDS_PER_HASH + DIGEST_WIRES];
	let step_of = |k: usize| hinted[k * HINT_WORDS_PER_HASH + DIGEST_WIRES + 1];
	let chain_ends: [[Wire; DIGEST_WIRES]; V] = std::array::from_fn(|i| {
		std::array::from_fn(|w| {
			hinted[NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH + i * DIGEST_WIRES + w]
		})
	});

	// One row per chain: its tip, its digit, its end. Every entry indexes this by its own chain.
	let table: Vec<Vec<Wire>> = (0..V)
		.map(|i| {
			let mut row = Vec::with_capacity(2 * DIGEST_WIRES + 1);
			row.extend_from_slice(&chain_tips[i]);
			row.push(digits[i]);
			row.extend_from_slice(&chain_ends[i]);
			row
		})
		.collect();
	let table_rows = table.iter().map(|row| row.as_slice()).collect::<Vec<_>>();

	// Which chains an entry can possibly belong to. A chain contributes at most
	// `CHAIN_LENGTH - 1` entries, of which at most `CHAIN_LENGTH - 2` can fall on one side of any
	// entry of its own. So of the `k` entries before this one, all but `CHAIN_LENGTH - 2` need
	// chains strictly below its own, and likewise above for the entries after it. That confines
	// each entry to about two thirds of the chains, and only that window has to be muxed over.
	let window = |k: usize| -> (usize, usize) {
		let per_chain = CHAIN_LENGTH - 1;
		let own_side = CHAIN_LENGTH - 2;
		let before = k.saturating_sub(own_side).div_ceil(per_chain);
		let after = (NUM_CHAIN_HASHES - 1 - k)
			.saturating_sub(own_side)
			.div_ceil(per_chain);
		(before, V - 1 - after)
	};

	// The hashes. Every entry's input is hinted rather than carried from the entry before it, so
	// they are independent and pair two to a core.
	let sub_position = |k: usize| builder.bxor(builder.shl(chain_of(k), W as u32), step_of(k));
	let mut outputs = Vec::with_capacity(NUM_CHAIN_HASHES);
	for pair in 0..NUM_CHAIN_HASHES / 2 {
		let (a, b) = (2 * pair, 2 * pair + 1);
		let (in_a, in_b) = (input_of(a), input_of(b));
		let digests = circuit_tweak_hash_2x(
			builder,
			public_param,
			TWEAK_TYPE_CHAIN,
			[sub_position(a), sub_position(b)],
			epoch,
			[&in_a, &in_b],
		);
		outputs.extend_from_slice(&digests);
	}
	if NUM_CHAIN_HASHES % 2 == 1 {
		let k = NUM_CHAIN_HASHES - 1;
		outputs.push(circuit_tweak_hash(
			builder,
			public_param,
			TWEAK_TYPE_CHAIN,
			sub_position(k),
			epoch,
			&input_of(k),
		));
	}

	let one = builder.add_constant_64(1);
	let last_step = builder.add_constant_64((CHAIN_LENGTH - 2) as u64);
	let never = builder.add_constant(Word::ZERO);
	let always = builder.add_constant(Word::ALL_ONE);

	for k in 0..NUM_CHAIN_HASHES {
		let b = builder.subcircuit(format!("chain_hash[{k}]"));
		let (chain, step, input) = (chain_of(k), step_of(k), input_of(k));

		// The window is asserted rather than merely implied: the mux reads only the low bits of
		// its selector, so a chain outside the window would alias onto a row that is not its own.
		let (lowest, highest) = window(k);
		let lowest_wire = b.add_constant_64(lowest as u64);
		b.assert_true("chain_at_least_window", b.icmp_ule(lowest_wire, chain));
		b.assert_true("chain_at_most_window", b.icmp_ule(chain, b.add_constant_64(highest as u64)));

		let zero = b.add_constant_64(0);
		let offset = b.isub_bin_bout(chain, lowest_wire, zero).0;
		let row = multi_wire_multiplex(&b, &table_rows[lowest..=highest], offset);
		let tip: [Wire; DIGEST_WIRES] = std::array::from_fn(|w| row[w]);
		let digit = row[DIGEST_WIRES];
		let end: [Wire; DIGEST_WIRES] = std::array::from_fn(|w| row[DIGEST_WIRES + 1 + w]);

		// An entry either continues the one before it or opens a new chain. The chain field is
		// what says which, and it never decreases, so a chain's entries stay contiguous.
		let continues = if k == 0 {
			never
		} else {
			b.assert_true("chain_non_decreasing", b.icmp_ule(chain_of(k - 1), chain));
			b.icmp_eq(chain, chain_of(k - 1))
		};

		if k > 0 {
			let expected_step = b.iadd(step_of(k - 1), one).0;
			b.assert_eq("step_advances", b.select(continues, step, expected_step), expected_step);
			for w in 0..DIGEST_WIRES {
				let previous = outputs[k - 1][w];
				b.assert_eq(
					format!("input_continues[{w}]"),
					b.select(continues, input[w], previous),
					previous,
				);
			}
		}

		// Opening a chain: start at the digit, from the signature's tip for that chain.
		let starts = b.bnot(continues);
		b.assert_eq("starts_at_digit", b.select(starts, step, digit), digit);
		for w in 0..DIGEST_WIRES {
			b.assert_eq(
				format!("starts_from_tip[{w}]"),
				b.select(starts, input[w], tip[w]),
				tip[w],
			);
		}

		// Closing a chain: end at the last step, and hand the chain its public-key end.
		let ends = if k + 1 == NUM_CHAIN_HASHES {
			always
		} else {
			b.bnot(b.icmp_eq(chain_of(k + 1), chain))
		};
		b.assert_eq("ends_at_last_step", b.select(ends, step, last_step), last_step);
		for w in 0..DIGEST_WIRES {
			b.assert_eq(
				format!("ends_at_chain_end[{w}]"),
				b.select(ends, outputs[k][w], end[w]),
				end[w],
			);
		}
	}

	// A chain at the last digit owes no hashes and so has no entry to close it: its end is the
	// tip the signature already gave.
	let empty_digit = builder.add_constant_64((CHAIN_LENGTH - 1) as u64);
	for i in 0..V {
		let empty = builder.icmp_eq(digits[i], empty_digit);
		for w in 0..DIGEST_WIRES {
			builder.assert_eq(
				format!("empty_chain_end[{i}][{w}]"),
				builder.select(empty, chain_ends[i][w], chain_tips[i][w]),
				chain_tips[i][w],
			);
		}
	}

	chain_ends
}

/// In-circuit form of [`wots_public_key_hash`].
pub fn circuit_wots_public_key_hash(
	builder: &CircuitBuilder,
	public_param: &[Wire; PUBLIC_PARAM_WIRES],
	epoch: Wire,
	chain_ends: &[[Wire; DIGEST_WIRES]; V],
) -> [Wire; DIGEST_WIRES] {
	let payload = chain_ends.iter().flatten().copied().collect::<Vec<_>>();
	let zero = builder.add_constant_64(0);
	circuit_tweak_hash(builder, public_param, TWEAK_TYPE_WOTS_PK, zero, epoch, &payload)
}

#[cfg(test)]
mod tests {
	use binius_core::Word;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::hash_based_sig::PUBLIC_PARAM_LEN;

	/// A signature at `epoch`: random chain preimages walked to the digits the message encodes to.
	struct TestSignature {
		public_param: PublicParam,
		message: Message,
		randomness: Randomness,
		encoding: [u8; V],
		chain_tips: [Digest; V],
		chain_ends: [Digest; V],
	}

	impl TestSignature {
		fn generate(rng: &mut StdRng, epoch: u32) -> Self {
			let mut public_param = [0u8; PUBLIC_PARAM_LEN];
			rng.fill_bytes(&mut public_param);
			let mut message = [0u8; MESSAGE_LEN];
			rng.fill_bytes(&mut message);

			let (randomness, encoding) =
				find_randomness_for_wots_encoding(&message, epoch, &public_param, rng);

			// A signature's chain tip is the secret preimage walked as far as its digit; the
			// verifier walks the rest.
			let mut pre_images = [[0u8; DIGEST_LEN]; V];
			for pre_image in pre_images.iter_mut() {
				rng.fill_bytes(pre_image);
			}
			let chain_tips: [Digest; V] = std::array::from_fn(|i| {
				iterate_hash(&pre_images[i], encoding[i] as usize, &public_param, epoch, i, 0)
			});
			let chain_ends = recover_public_key(&chain_tips, &encoding, epoch, &public_param);

			Self {
				public_param,
				message,
				randomness,
				encoding,
				chain_tips,
				chain_ends,
			}
		}
	}

	#[test]
	fn encoding_is_valid_by_construction() {
		let mut rng = StdRng::seed_from_u64(0);
		let sig = TestSignature::generate(&mut rng, 7);
		assert_eq!(sig.encoding.iter().map(|&e| e as usize).sum::<usize>(), TARGET_SUM);
		assert!(sig.encoding.iter().all(|&e| (e as usize) < CHAIN_LENGTH));
	}

	#[test]
	fn the_fixture_covers_empty_and_walked_chains() {
		// `circuit_recovers_the_public_key` only exercises the empty-chain path if some chain is
		// actually empty. With the target sum putting the mean digit at 4.64 that is the common
		// case, but it is worth failing loudly if a fixture ever stops covering it.
		let mut rng = StdRng::seed_from_u64(1);
		let sig = TestSignature::generate(&mut rng, 12345);
		assert!(
			sig.encoding.iter().any(|&e| e as usize == CHAIN_LENGTH - 1),
			"no chain is empty, so the zero-hash path goes unchecked"
		);
		assert!(
			sig.encoding
				.iter()
				.any(|&e| (e as usize) < CHAIN_LENGTH - 1),
			"every chain is empty, so no chain hash is walked"
		);
	}

	#[test]
	fn a_chain_end_is_its_tip_at_the_last_digit() {
		// The one chain the verifier never advances.
		let pp = [4u8; PUBLIC_PARAM_LEN];
		let tip = [9u8; DIGEST_LEN];
		assert_eq!(iterate_hash(&tip, CHAIN_LENGTH - 1 - (CHAIN_LENGTH - 1), &pp, 3, 0, 7), tip);
	}

	/// Builds the encode-and-walk circuit, populates it from `sig`, and returns the result of
	/// checking the constraint system.
	fn run(sig: &TestSignature, epoch: u32) -> Result<(), String> {
		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let randomness_w: [Wire; RANDOMNESS_WIRES] = std::array::from_fn(|_| b.add_witness());
		let tips_w: [[Wire; DIGEST_WIRES]; V] =
			std::array::from_fn(|_| std::array::from_fn(|_| b.add_witness()));
		let leaf_w: [Wire; DIGEST_WIRES] = std::array::from_fn(|_| b.add_inout());

		let digits = circuit_wots_encode(&b, &param_w, epoch_w, &message_w, &randomness_w);
		let ends = circuit_recover_public_key(&b, &param_w, epoch_w, &tips_w, &digits);
		let leaf = circuit_wots_public_key_hash(&b, &param_w, epoch_w, &ends);
		b.assert_eq_v("leaf", leaf, leaf_w);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &sig.public_param);
		w[epoch_w] = Word::from_u64(epoch as u64);
		w.pack_bytes_le(&message_w, &sig.message);
		w.pack_bytes_le(&randomness_w, &sig.randomness);
		for (wires, tip) in tips_w.iter().zip(&sig.chain_tips) {
			w.pack_bytes_le(wires, tip);
		}
		w.pack_bytes_le(&leaf_w, &wots_public_key_hash(&sig.public_param, epoch, &sig.chain_ends));

		circuit
			.populate_wire_witness(&mut w)
			.map_err(|e| format!("populate: {e:?}"))?;
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.map_err(|e| format!("verify: {e:?}"))
	}

	#[test]
	fn circuit_recovers_the_public_key() {
		let mut rng = StdRng::seed_from_u64(1);
		let epoch = 12345;
		let sig = TestSignature::generate(&mut rng, epoch);
		run(&sig, epoch).unwrap();
	}

	#[test]
	fn circuit_digits_match_the_reference_encoding() {
		let mut rng = StdRng::seed_from_u64(2);
		let epoch = 9;
		let sig = TestSignature::generate(&mut rng, epoch);

		let b = CircuitBuilder::new();
		let param_w: [Wire; PUBLIC_PARAM_WIRES] = std::array::from_fn(|_| b.add_inout());
		let epoch_w = b.add_inout();
		let message_w: [Wire; MESSAGE_WIRES] = std::array::from_fn(|_| b.add_inout());
		let randomness_w: [Wire; RANDOMNESS_WIRES] = std::array::from_fn(|_| b.add_inout());
		let digits = circuit_wots_encode(&b, &param_w, epoch_w, &message_w, &randomness_w);
		let expected: [Wire; V] = std::array::from_fn(|_| b.add_inout());
		b.assert_eq_v("digits", digits, expected);

		let circuit = b.build();
		let mut w = circuit.new_witness_filler();
		w.pack_bytes_le(&param_w, &sig.public_param);
		w[epoch_w] = Word::from_u64(epoch as u64);
		w.pack_bytes_le(&message_w, &sig.message);
		w.pack_bytes_le(&randomness_w, &sig.randomness);
		for (wire, &digit) in expected.iter().zip(&sig.encoding) {
			w[*wire] = Word::from_u64(digit as u64);
		}

		circuit.populate_wire_witness(&mut w).unwrap();
		circuit
			.constraint_system()
			.verify(&w.into_value_vec())
			.unwrap();
	}

	#[test]
	fn circuit_rejects_randomness_that_does_not_encode() {
		let mut rng = StdRng::seed_from_u64(3);
		let epoch = 4;
		let mut sig = TestSignature::generate(&mut rng, epoch);

		// Any randomness the grinder did not settle on fails one of the two conditions, so the
		// encoding constraints have no satisfying witness.
		let mut bad = sig.randomness;
		bad[0] ^= 0xFF;
		assert!(
			wots_encode(&sig.message, epoch, &sig.public_param, &bad).is_none(),
			"the tampered randomness happened to encode validly; pick another"
		);
		sig.randomness = bad;
		assert!(run(&sig, epoch).is_err(), "an invalid encoding must not verify");
	}

	#[test]
	fn circuit_rejects_a_tampered_chain_tip() {
		let mut rng = StdRng::seed_from_u64(4);
		let epoch = 4;
		let mut sig = TestSignature::generate(&mut rng, epoch);
		sig.chain_tips[0][0] ^= 0xFF;
		assert!(run(&sig, epoch).is_err(), "a tampered tip must not reach the public key");
	}

	#[test]
	fn circuit_rejects_a_signature_from_another_epoch() {
		let mut rng = StdRng::seed_from_u64(5);
		let epoch = 4;
		let sig = TestSignature::generate(&mut rng, epoch);
		// Every tweak carries the epoch, so the chains and the encoding both move with it.
		assert!(run(&sig, epoch + 1).is_err(), "an epoch it was not signed at must not verify");
	}
}
