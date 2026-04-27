// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{fixed_byte_vec::ByteVec, slice};

/// Asserts that a JSON string contains specific attribute values.
///
/// Validates that `json` (with actual length `len_bytes`) contains each `(name, value)` pair as
/// `"<name>":"<value>"` where the value matches the bytes of the supplied [`ByteVec`].
///
/// This circuit makes some strong assumptions, in particular:
///
/// 1. The input is a valid JSON object. No effort is put into checking that or any other properties
///    such as duplicate keys.
/// 2. No whitespace handling.
/// 3. The attributes of interest are strings only. Matching arrays or objects as values is not
///    supported.
///
/// ❗️ At this point, the circuit does not check that it did not get multiple attributes. For
/// example, imagine there are two attributes "sub" and "nonce" where one is right next to the
/// other. A prover can get away with providing the attribute "sub" but the end quote of the nonce.
///
/// # Arguments
/// * `b` - Circuit builder
/// * `len_bytes` - Wire for actual JSON size in bytes
/// * `json` - JSON input array packed as words (8 bytes per word)
/// * `attributes` - Slice of `(name, value)` pairs to verify, where `value` is a `ByteVec` carrying
///   the expected bytes and runtime length.
pub fn jwt_claims(
	b: &CircuitBuilder,
	len_bytes: Wire,
	json: &[Wire],
	attributes: &[(&str, &ByteVec)],
) {
	// For each attribute, we need to:
	// 1. Find the pattern "name":" in the JSON
	// 2. Extract the string value between the quotes
	// 3. Verify it matches the expected value
	let max_len_bytes = json.len() << 3;
	let too_long = b.icmp_ult(b.add_constant_64(max_len_bytes as u64), len_bytes);
	b.assert_false("length check", too_long);

	for (attr_idx, &(name, value)) in attributes.iter().enumerate() {
		let b = b.subcircuit(format!("attr[ix={attr_idx}, name={name}]"));

		// Build the search pattern: "name":"
		let pattern = format!("\"{name}\":\"");
		let pattern_bytes = pattern.as_bytes();
		let pattern_len = pattern_bytes.len();

		// ---- Pattern matching algorithm
		//
		// We search for the pattern "name":" in the JSON by checking every possible
		// starting position. Since we can't break out of loops in circuits, we check
		// all positions and use masking to track where we found matches.
		//
		// Variables:
		// - found_position: the position where we found the pattern (0 if not found yet)
		// - any_found: becomes msb-true when we find the pattern anywhere
		let zero = b.add_constant(Word::ZERO);
		let mut value_start = zero;
		let mut found_start = zero;

		// Check each possible starting position
		for start_pos in 0..max_len_bytes.saturating_sub(pattern_len) {
			let b = b.subcircuit(format!("start_pos[{start_pos}]"));

			// Check if this position could contain the full pattern
			let end_wire = b.add_constant(Word((start_pos + pattern_len) as u64));

			// Verify position is within JSON bounds
			let within_bounds = b.icmp_ult(end_wire, len_bytes);
			// strict inequality is safe, because there will also be a closing "
			let mut matches_here = within_bounds;

			// Check each byte of the pattern
			for (i, &expected_byte) in pattern_bytes.iter().enumerate() {
				let byte_pos = start_pos + i;
				let word_idx = byte_pos / 8;
				let byte_offset = byte_pos % 8;

				let actual_byte = b.extract_byte(json[word_idx], byte_offset as u32);
				let expected = b.add_constant(Word(expected_byte as u64));
				let byte_matches = b.icmp_eq(actual_byte, expected);
				matches_here = b.band(matches_here, byte_matches);
			}

			// If we found a match here, remember this position
			value_start = b.select(matches_here, end_wire, value_start);
			found_start = b.bor(found_start, matches_here);
		}

		// Assert that we found the pattern (found_start should be msb-true)
		b.assert_true("attr_found".to_string(), found_start);

		// ---- Find value terminator
		//
		// Search for the terminator that marks the end of the attribute value.
		// Valid terminators are: " (closing quote), , (comma), or } (closing brace)
		// We scan all positions starting from value_start and use masking to
		// remember where we found a terminator.
		let mut value_end = zero;
		let mut found_end = zero;
		let quote = b.add_constant_zx_8(b'"');
		let comma = b.add_constant_zx_8(b',');
		let close_brace = b.add_constant_zx_8(b'}');
		// i actually don't grok why these latter two are valid terminators.

		for pos in 0..max_len_bytes {
			let b = b.subcircuit(format!("find_terminator[{pos}]"));

			let pos_wire = b.add_constant(Word(pos as u64));
			let within_bounds = b.icmp_ult(pos_wire, len_bytes);

			// Check if this position is after value_start
			// For empty strings, the closing quote is at value_start
			let at_or_after_start = b.bnot(b.icmp_ult(pos_wire, value_start));
			let not_found_yet = b.bnot(found_end);
			let should_check = b.band(b.band(at_or_after_start, within_bounds), not_found_yet);

			// Extract byte at this position
			let word_idx = pos / 8;
			let byte_offset = pos % 8;

			let byte_at_pos = b.extract_byte(json[word_idx], byte_offset as u32);

			// Check if this byte is any of the valid terminators
			let is_quote = b.icmp_eq(byte_at_pos, quote);
			let is_comma = b.icmp_eq(byte_at_pos, comma);
			let is_close_brace = b.icmp_eq(byte_at_pos, close_brace);

			// It's a terminator if it's any of the three
			let is_terminator = b.bor(b.bor(is_quote, is_comma), is_close_brace);

			let found_here = b.band(should_check, is_terminator);

			// If we found a terminator here, remember this position
			// When found_here is all-1s, include pos_wire in value_end
			// When found_here is all-0s, masked_pos is 0 and OR leaves value_end unchanged
			value_end = b.select(found_here, pos_wire, value_end);
			found_end = b.bor(found_end, found_here);
		}

		// Assert that we found a terminator (found_end should be all-1s)
		b.assert_true("attr_terminator_found", found_end);

		// Calculate value length: value_end - value_start
		// Since circuits don't have a subtraction operation, we use two's complement:
		// a - b = a + (~b + 1), where ~b is bitwise NOT of b
		let (value_length, _borrow) = b.isub_bin_bout(value_end, value_start, zero);

		// Verify the length matches expected
		b.assert_eq("attr_length", value_length, value.len_bytes);

		// Extract the value bytes from the JSON and assert they match the caller-supplied value.
		let extracted =
			slice::slice(&b, len_bytes, value_length, json, value_start, value.data.len());
		for (i, (&a, &e)) in extracted.iter().zip(&value.data).enumerate() {
			b.assert_eq(format!("attr_value[{i}]"), a, e);
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::{CircuitBuilder, util::pack_bytes_into_wires_le};

	use super::{ByteVec, Wire, Word, jwt_claims};

	/// Helper that allocates a `ByteVec` of `n_words` inout wires.
	fn alloc_byte_vec(b: &CircuitBuilder, n_words: usize) -> ByteVec {
		ByteVec {
			len_bytes: b.add_inout(),
			data: (0..n_words).map(|_| b.add_inout()).collect(),
		}
	}

	#[test]
	fn test_single_attribute() {
		let b = CircuitBuilder::new();

		let len_json = b.add_witness();
		let json: Vec<Wire> = (0..32).map(|_| b.add_witness()).collect();

		let sub = alloc_byte_vec(&b, 2);

		jwt_claims(&b, len_json, &json, &[("sub", &sub)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		let json_str = r#"{"sub":"1234567890","iss":"google.com"}"#;

		// Populate inputs
		filler[len_json] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected value
		sub.populate_len_bytes(&mut filler, 10);
		sub.populate_data(&mut filler, b"1234567890");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_multiple_attributes() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..32).map(|_| b.add_witness()).collect();

		let sub = alloc_byte_vec(&b, 2);
		let iss = alloc_byte_vec(&b, 4);
		let aud = alloc_byte_vec(&b, 2);

		jwt_claims(&b, len_bytes, &json, &[("sub", &sub), ("iss", &iss), ("aud", &aud)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// Test JSON with all attributes
		let json_str =
			r#"{"sub":"1234567890","iss":"google.com","aud":"4074087","iat":1676415809}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected values
		sub.populate_len_bytes(&mut filler, 10);
		sub.populate_data(&mut filler, b"1234567890");

		iss.populate_len_bytes(&mut filler, 10);
		iss.populate_data(&mut filler, b"google.com");

		aud.populate_len_bytes(&mut filler, 7);
		aud.populate_data(&mut filler, b"4074087");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_attribute_not_found() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..16).map(|_| b.add_witness()).collect();

		let missing = alloc_byte_vec(&b, 2);

		jwt_claims(&b, len_bytes, &json, &[("missing", &missing)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// JSON without the required attribute
		let json_str = r#"{"sub":"1234567890","iss":"google.com"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected value (won't be found)
		missing.populate_len_bytes(&mut filler, 5);
		missing.populate_data(&mut filler, b"value");

		// This should fail because "missing" attribute is not in the JSON
		let result = circuit.populate_wire_witness(&mut filler);
		assert!(result.is_err());
	}

	#[test]
	fn test_wrong_value() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..16).map(|_| b.add_witness()).collect();

		let sub = alloc_byte_vec(&b, 2);

		jwt_claims(&b, len_bytes, &json, &[("sub", &sub)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// Test JSON
		let json_str = r#"{"sub":"1234567890","iss":"google.com"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate wrong expected value
		sub.populate_len_bytes(&mut filler, 10);
		sub.populate_data(&mut filler, b"9876543210");

		// This should fail because the value doesn't match
		let result = circuit.populate_wire_witness(&mut filler);
		assert!(result.is_err());
	}

	#[test]
	fn test_attributes_in_different_order() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..32).map(|_| b.add_witness()).collect();

		let aud = alloc_byte_vec(&b, 16 / 8);
		let sub = alloc_byte_vec(&b, 16 / 8);

		jwt_claims(&b, len_bytes, &json, &[("aud", &aud), ("sub", &sub)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// JSON with attributes in different order
		let json_str =
			r#"{"iss":"google.com","sub":"1234567890","email":"test@example.com","aud":"4074087"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected values
		aud.populate_len_bytes(&mut filler, 7);
		aud.populate_data(&mut filler, b"4074087");

		sub.populate_len_bytes(&mut filler, 10);
		sub.populate_data(&mut filler, b"1234567890");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_empty_string_value() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..16).map(|_| b.add_witness()).collect();

		let empty = alloc_byte_vec(&b, 1);

		jwt_claims(&b, len_bytes, &json, &[("empty", &empty)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// JSON with empty string value
		let json_str = r#"{"empty":"","sub":"123"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected empty value
		empty.populate_len_bytes(&mut filler, 0);
		empty.populate_data(&mut filler, b"");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_special_characters() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..32).map(|_| b.add_witness()).collect();

		let email = alloc_byte_vec(&b, 4);
		let nonce = alloc_byte_vec(&b, 4);

		jwt_claims(&b, len_bytes, &json, &[("email", &email), ("nonce", &nonce)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// JSON with special characters
		let json_str = r#"{"email":"john.doe@gmail.com","nonce":"7-VU9fuWeWtgDLHmVJ2UtRrine8"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected values
		email.populate_len_bytes(&mut filler, 18);
		email.populate_data(&mut filler, b"john.doe@gmail.com");

		nonce.populate_len_bytes(&mut filler, 27);
		nonce.populate_data(&mut filler, b"7-VU9fuWeWtgDLHmVJ2UtRrine8");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	#[test]
	fn test_last_attribute_no_comma() {
		let b = CircuitBuilder::new();

		let len_bytes = b.add_witness();
		let json: Vec<Wire> = (0..16).map(|_| b.add_witness()).collect();

		let iss = alloc_byte_vec(&b, 16 / 8);
		let last = alloc_byte_vec(&b, 16 / 8);

		jwt_claims(&b, len_bytes, &json, &[("iss", &iss), ("last", &last)]);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();

		// JSON where the last attribute has no comma after it (terminated by })
		let json_str = r#"{"iss":"example.com","last":"value123"}"#;

		// Populate inputs
		filler[len_bytes] = Word(json_str.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &json, json_str.as_bytes());

		// Populate expected values
		iss.populate_len_bytes(&mut filler, 11);
		iss.populate_data(&mut filler, b"example.com");

		last.populate_len_bytes(&mut filler, 8);
		last.populate_data(&mut filler, b"value123");

		circuit.populate_wire_witness(&mut filler).unwrap();

		// Verify constraints
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}
}
