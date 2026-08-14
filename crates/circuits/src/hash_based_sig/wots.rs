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

/// Words the hint emits per chain hash: the input digest, then the chain, the step and the digit.
const HINT_WORDS_PER_HASH: usize = DIGEST_WIRES + 3;

/// Words the hint reads.
const HINT_INPUTS: usize = PUBLIC_PARAM_WIRES + 1 + V + V * DIGEST_WIRES;

/// Words the hint writes: one entry per chain hash, then where each chain's last entry sits.
const HINT_OUTPUTS: usize = NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH + V;

/// Computes the chain hashes a verifier actually walks, and where each chain's last one sits.
///
/// The verifier's work is the concatenation of the chain tails: chain `i` contributes its steps
/// from `digit_i` up to `CHAIN_LENGTH - 2`, and the tails run in chain order. How long each tail
/// is depends on the digits, so the list cannot be laid out at circuit construction time — it is
/// hinted here and pinned by the constraints in [`circuit_recover_public_key`].
///
/// The offsets are hinted for the same reason and need no separate pinning: an offset is only ever
/// used to index the list, and what it lands on is checked.
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

		let (hashes, offsets) = outputs.split_at_mut(NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH);
		hashes.fill(Word::ZERO);
		offsets.fill(Word::ZERO);

		let mut written = 0;
		for i in 0..V {
			let digit = digits[i].as_u64() as usize;
			let mut current =
				bytes_from_words::<DIGEST_LEN>(&tips[i * DIGEST_WIRES..][..DIGEST_WIRES]);

			// A chain walks from its digit to the last step. A digit of `CHAIN_LENGTH - 1` walks
			// nothing, and a digit past that (an unsatisfiable witness) walks nothing either.
			for (step, position) in (digit..CHAIN_LENGTH - 1).enumerate() {
				// Digits that miss the target sum overrun the list. The encoding constraints
				// reject them, so stopping short here only has to avoid a panic.
				if written == NUM_CHAIN_HASHES {
					break;
				}
				let slot = &mut hashes[written * HINT_WORDS_PER_HASH..][..HINT_WORDS_PER_HASH];
				bytes_to_words(&current, &mut slot[..DIGEST_WIRES]);
				slot[DIGEST_WIRES] = Word::from_u64(i as u64);
				slot[DIGEST_WIRES + 1] = Word::from_u64(step as u64);
				slot[DIGEST_WIRES + 2] = Word::from_u64(digit as u64);

				current = chain_step(&public_param, epoch, i, position, &current);
				written += 1;
				// An empty chain leaves its offset at zero; nothing reads it.
				offsets[i] = Word::from_u64((written - 1) as u64);
			}
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
/// by constraints. Each entry carries its input digest, its chain, its digit, and the number of
/// hashes that chain has already done; its output is the hash, not a hinted value, so nothing has
/// to check it.
///
/// # What pins the list
///
/// Walking the list, every rule is local, which is what leaves the entries with nothing to look
/// up:
///
/// - `chain` only ever increases, so a chain's entries are one contiguous run.
/// - Within a run the step advances by one, the digit holds, and each input is the previous output.
/// - A run opens at step zero, and so does the list.
///
/// Then one lookup per chain finds that chain's last entry, and checks it belongs to this chain,
/// carries this chain's digit, and sits at the last position — `digit + step == CHAIN_LENGTH - 2`,
/// since the last hash of a chain starts one below the chain's end. Its output is the chain's end.
/// The offset is hinted; it needs no pinning of its own, because what it lands on is checked.
///
/// A run's length is therefore exactly `CHAIN_LENGTH - 1 - digit`, and the list is exactly
/// [`NUM_CHAIN_HASHES`] long, which the target sum makes the sum of those lengths. **No chain that
/// owes hashes can be missing a run**: if one were, the entries would not add up.
///
/// What a run starts *from* is left free. The tip is a hint, and verification only needs some
/// preimage that walks a chain of the right length onto the committed public key — a prover with
/// nothing to reveal would have to invert the hash to find one. A chain whose digit is
/// `CHAIN_LENGTH - 1` owes no hashes at all, and its end is simply the value the signature
/// revealed.
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
	let digit_of = |k: usize| hinted[k * HINT_WORDS_PER_HASH + DIGEST_WIRES + 2];
	let offset_of = |i: usize| hinted[NUM_CHAIN_HASHES * HINT_WORDS_PER_HASH + i];

	// Where a chain's last entry can sit. Chains before this one contribute at most
	// `CHAIN_LENGTH - 1` entries each and so do the chains after, which pins the last entry of
	// chain `i` into a window of the list — and only that window has to be muxed over.
	let window = |i: usize| -> (usize, usize) {
		let per_chain = CHAIN_LENGTH - 1;
		let earliest = (NUM_CHAIN_HASHES).saturating_sub((V - i) * per_chain);
		let latest = ((i + 1) * per_chain - 1).min(NUM_CHAIN_HASHES - 1);
		(earliest, latest)
	};

	// The hashes. Every entry's input is hinted rather than carried from the entry before it, so
	// they are independent and pair two to a core. A chain link's tweak names the position the
	// hash starts from, which is the chain's digit plus the hashes it has already done.
	let sub_position = |k: usize| {
		let position = builder.iadd(digit_of(k), step_of(k)).0;
		builder.bxor(builder.shl(chain_of(k), W as u32), position)
	};
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

	let zero = builder.add_constant_64(0);
	let one = builder.add_constant_64(1);

	// Walking the list: every rule here is local, which is what leaves the entries with nothing
	// to look up.
	for k in 0..NUM_CHAIN_HASHES {
		let b = builder.subcircuit(format!("chain_hash[{k}]"));
		let (chain, step) = (chain_of(k), step_of(k));

		let Some(previous) = k.checked_sub(1) else {
			// The list opens a chain, so it opens at its first hash.
			b.assert_eq("first_step_is_zero", step, zero);
			continue;
		};

		// An entry either continues the one before it or opens a new chain. The chain field is
		// what says which, and it only ever increases, so a chain's entries stay contiguous.
		let continues = b.icmp_eq(chain, chain_of(previous));
		let opens = b.bnot(continues);
		b.assert_true("chain_non_decreasing", b.icmp_ule(chain_of(previous), chain));

		// Continuing: one more hash of the same chain, on the value the last one produced.
		let next_step = b.iadd(step_of(previous), one).0;
		b.assert_eq("step_advances", b.select(continues, step, next_step), next_step);
		b.assert_eq(
			"digit_holds",
			b.select(continues, digit_of(k), digit_of(previous)),
			digit_of(previous),
		);
		for w in 0..DIGEST_WIRES {
			let carried = outputs[previous][w];
			b.assert_eq(
				format!("input_continues[{w}]"),
				b.select(continues, input_of(k)[w], carried),
				carried,
			);
		}

		// Opening: a fresh chain starts over at its first hash. What it starts *from* is left
		// free — the tip is a hint, and verification only needs some preimage that walks a chain
		// of the right length onto the committed public key.
		b.assert_eq("opens_at_zero", b.select(opens, step, zero), zero);
	}

	// One lookup per chain, into the window its last entry has to sit in. What the offset lands
	// on is checked, so the offset itself needs no pinning.
	std::array::from_fn(|i| {
		let b = builder.subcircuit(format!("chain_end[{i}]"));
		let (earliest, latest) = window(i);
		let entries = (earliest..=latest)
			.map(|k| {
				vec![
					chain_of(k),
					step_of(k),
					digit_of(k),
					outputs[k][0],
					outputs[k][1],
				]
			})
			.collect::<Vec<_>>();
		let rows = entries.iter().map(|e| e.as_slice()).collect::<Vec<_>>();

		let earliest_wire = b.add_constant_64(earliest as u64);
		let index = b.isub_bin_bout(offset_of(i), earliest_wire, zero).0;
		let found = multi_wire_multiplex(&b, &rows, index);
		let (chain, step, digit) = (found[0], found[1], found[2]);
		let end: [Wire; DIGEST_WIRES] = std::array::from_fn(|w| found[DIGEST_WIRES + 1 + w]);

		// A chain at the last digit owes no hashes, so it has no entry to find: its end is the
		// value the signature already revealed, and nothing about the lookup is asserted.
		let walks = b.bnot(b.icmp_eq(digits[i], b.add_constant_64((CHAIN_LENGTH - 1) as u64)));

		let expected_chain = b.add_constant_64(i as u64);
		b.assert_eq("chain_is_this_one", b.select(walks, chain, expected_chain), expected_chain);
		b.assert_eq("digit_is_the_encoding", b.select(walks, digit, digits[i]), digits[i]);

		// The last hash of a chain starts one below the end of the chain, so its position —
		// the digit plus the hashes done before it — is `CHAIN_LENGTH - 2`.
		let last_position = b.add_constant_64((CHAIN_LENGTH - 2) as u64);
		let position = b.iadd(digit, step).0;
		b.assert_eq(
			"ends_at_the_last_position",
			b.select(walks, position, last_position),
			last_position,
		);

		std::array::from_fn(|w| b.select(walks, end[w], chain_tips[i][w]))
	})
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
