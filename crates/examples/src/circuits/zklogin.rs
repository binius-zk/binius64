// Copyright 2025 Irreducible Inc.
use anyhow::{Result, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD};
use binius_circuits::{
	base64::base64_url_safe, concat::concat, fixed_byte_vec::ByteVec, jwt_claims::jwt_claims,
	rs256::Rs256Verify, sha256::Sha256 as Sha256Circuit, slice::create_byte_mask,
};
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller, util::pack_bytes_into_wires_le};
use clap::Args;
use jwt_simple::prelude::*;
use rand::prelude::*;
use sha2::{Digest, Sha256};

use crate::ExampleCircuit;

/// The configuration of the ZKLogin circuit.
///
/// Picking the numbers are a tradeoff. Picking a large number will require a larger circuit and
/// thus more proving time. Picking a small number may make some statements unprovable.
#[derive(Debug, Clone)]
pub struct Config {
	/// Maximum length in wires of the base64 decoded JWT header. Must be a multiple of 8.
	pub max_len_json_jwt_header: usize,
	/// Maximum length in wires of the base64 decoded JWT payload. Must be a multiple of 8.
	pub max_len_json_jwt_payload: usize,
	/// Maximum length in wires of the base64 decoded JWT signature. Must be a multiple of 8.
	pub max_len_jwt_signature: usize,
	pub max_len_jwt_sub: usize,
	pub max_len_jwt_aud: usize,
	pub max_len_jwt_iss: usize,
	pub max_len_salt: usize,
	pub max_len_nonce_r: usize,
	pub max_len_t_max: usize,
}

impl Default for Config {
	fn default() -> Self {
		Self {
			max_len_json_jwt_header: 33,
			max_len_json_jwt_payload: 63,
			max_len_jwt_signature: 33,
			max_len_jwt_sub: 9,
			max_len_jwt_aud: 9,
			max_len_jwt_iss: 9,
			max_len_salt: 9,
			max_len_nonce_r: 6,
			max_len_t_max: 6,
		}
	}
}

impl Config {
	pub fn max_len_base64_jwt_header(&self) -> usize {
		self.max_len_json_jwt_header.div_ceil(3) * 4
	}

	pub fn max_len_base64_jwt_payload(&self) -> usize {
		self.max_len_json_jwt_payload.div_ceil(3) * 4
	}

	pub fn max_len_base64_jwt_signature(&self) -> usize {
		self.max_len_jwt_signature.div_ceil(3) * 4
	}
}

/// A circuit that implements zk login.
pub struct ZkLogin {
	/// The sub claim value
	pub sub: ByteVec,
	/// The aud claim value
	pub aud: ByteVec,
	/// The iss claim value
	pub iss: ByteVec,
	/// The salt value
	pub salt: ByteVec,
	/// The zkaddr (SHA256 hash of concat(sub, aud, iss, salt))
	pub zkaddr: [Wire; 4],
	/// The SHA256 circuit for zkaddr verification
	pub zkaddr_sha256: Sha256Circuit,
	/// Inout-allocated `(alg, typ)` ByteVecs whose `len_bytes` wires are filled by
	/// [`Self::populate_jwt_header_attributes`].
	pub jwt_header_attrs: [ByteVec; 2],
	/// The subcircuit that verifies the RS256 signature in the JWT.
	pub jwt_signature_verify: Rs256Verify,
	/// The JWT header
	pub base64_jwt_header: ByteVec,
	/// The JWT payload
	pub base64_jwt_payload: ByteVec,
	/// The JWT signature
	pub base64_jwt_signature: ByteVec,
	/// The decoded JWT header
	pub jwt_header: ByteVec,
	/// The decoded jwt_payload
	pub jwt_payload: ByteVec,
	/// The decoded jwt_signature (264 bytes for Base64, little-endian packing)
	pub jwt_signature: ByteVec,
	/// The base64 encoded nonce
	pub base64_jwt_payload_nonce: [Wire; 6],
	/// The SHA256 circuit for nonce verification
	pub nonce_sha256: Sha256Circuit,
	/// The nonce value (32 bytes SHA256 hash)
	pub nonce: [Wire; 4],
	/// The vk_u public key (32 bytes)
	pub vk_u: [Wire; 4],
	/// The t_max value
	pub t_max: ByteVec,
	/// The nonce_r value
	pub nonce_r: ByteVec,
}

impl ZkLogin {
	pub fn new(b: &mut CircuitBuilder, config: Config) -> Self {
		let sub = ByteVec::new_inout(b, config.max_len_jwt_sub);
		let aud = ByteVec::new_inout(b, config.max_len_jwt_aud);
		let iss = ByteVec::new_inout(b, config.max_len_jwt_iss);
		let salt = ByteVec::new_inout(b, config.max_len_salt);

		let base64_jwt_header = ByteVec::new_inout(b, config.max_len_base64_jwt_header());
		let base64_jwt_payload = ByteVec::new_inout(b, config.max_len_base64_jwt_payload());
		let base64_jwt_signature = ByteVec::new_inout(b, config.max_len_base64_jwt_signature());

		let jwt_header = ByteVec::new_inout(b, config.max_len_json_jwt_header);
		let jwt_payload = ByteVec::new_witness(b, config.max_len_json_jwt_payload);
		let jwt_signature = ByteVec::new_witness(b, config.max_len_jwt_signature);

		let t_max = ByteVec::new_inout(b, config.max_len_t_max);
		let nonce_r = ByteVec::new_witness(b, config.max_len_nonce_r);

		let zkaddr: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		let vk_u: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
		let nonce: [Wire; 4] = std::array::from_fn(|_| b.add_witness());

		// The base64 encoded nonce in the JWT payload. This must have
		// 6 wires = 48 bytes to accommodate the 43-byte base64 nonce with padding.
		let base64_jwt_payload_nonce: [Wire; 6] = std::array::from_fn(|_| b.add_witness());

		// RSA modulus as public input (256 bytes for 2048-bit RSA)
		let rsa_modulus = ByteVec::new_inout(b, 32);

		// Decode JWT.
		// 1. header
		// 2. payload
		// 3. signature

		base64_url_safe(
			&b.subcircuit("base64_check_header"),
			&jwt_header.data,
			&base64_jwt_header.data,
			jwt_header.len_bytes,
		);
		base64_url_safe(
			&b.subcircuit("base64_check_payload"),
			&jwt_payload.data,
			&base64_jwt_payload.data,
			jwt_payload.len_bytes,
		);
		base64_url_safe(
			&b.subcircuit("base64_check_signature"),
			&jwt_signature.data,
			&base64_jwt_signature.data,
			jwt_signature.len_bytes,
		);

		// We need to check
		//
		// X = concat(JWT.sub, JWT.aud, JWT.iss, salt)
		// assert zkaddr == SHA256(X)
		let max_len_zkaddr_preimage = config.max_len_jwt_sub
			+ config.max_len_jwt_aud
			+ config.max_len_jwt_iss
			+ config.max_len_salt;

		// Create SHA256 verification for zkaddr first
		let zkaddr_preimage_len_bytes = b.add_witness();
		let zkaddr_sha256_message: Vec<Wire> = (0..max_len_zkaddr_preimage)
			.map(|_| b.add_witness())
			.collect();
		let zkaddr_sha256 = Sha256Circuit::new(
			&b.subcircuit("zkaddr_sha256"),
			zkaddr_preimage_len_bytes,
			zkaddr,
			zkaddr_sha256_message,
		);

		let zkaddr_preimage_le_wires = zkaddr_sha256.message_to_le_wires(b);
		let zkaddr_joined_words = max_len_zkaddr_preimage;
		let zkaddr_joined_le = zkaddr_preimage_le_wires[..zkaddr_joined_words].to_vec();

		// Compute the concatenation and assert it equals the SHA-256 message LE wires.
		let zkaddr_concat_b = b.subcircuit("zkaddr_preimage_concat");
		let zkaddr_concat = concat(
			&zkaddr_concat_b,
			&[
				ByteVec {
					data: sub.data.clone(),
					len_bytes: sub.len_bytes,
				},
				ByteVec {
					data: aud.data.clone(),
					len_bytes: aud.len_bytes,
				},
				ByteVec {
					data: iss.data.clone(),
					len_bytes: iss.len_bytes,
				},
				ByteVec {
					data: salt.data.clone(),
					len_bytes: salt.len_bytes,
				},
			],
			Some(zkaddr_joined_le.len()),
		);
		zkaddr_concat_b.assert_eq("len", zkaddr_concat.len_bytes, zkaddr_preimage_len_bytes);
		assert_sha256_message_eq_concat(&zkaddr_concat_b, &zkaddr_concat, &zkaddr_joined_le);

		// We need to check:
		//
		// nonce_preimage = concat(vk_u, T_max, r) where vk_u is a public key
		// assert nonce = SHA256(nonce_preimage)
		// assert nonce = base64_decode(base64_jwt_payload_nonce)
		let max_len_nonce_preimage = 4 + config.max_len_t_max + config.max_len_nonce_r;

		// Create SHA256 verification for nonce first
		let nonce_preimage_len_bytes = b.add_witness();
		let nonce_sha256_message: Vec<Wire> = (0..max_len_nonce_preimage)
			.map(|_| b.add_witness())
			.collect();
		let nonce_sha256 = Sha256Circuit::new(
			&b.subcircuit("nonce_sha256"),
			nonce_preimage_len_bytes,
			nonce,
			nonce_sha256_message,
		);

		let nonce_preimage_le_wires = nonce_sha256.message_to_le_wires(b);
		let nonce_joined_words = max_len_nonce_preimage;
		let nonce_joined_le = nonce_preimage_le_wires[..nonce_joined_words].to_vec();
		let nonce_concat_b = b.subcircuit("nonce_preimage_concat");
		let nonce_concat = concat(
			&nonce_concat_b,
			&[
				ByteVec {
					data: vk_u.to_vec(),
					len_bytes: nonce_concat_b.add_constant_64(32),
				},
				ByteVec {
					data: t_max.data.clone(),
					len_bytes: t_max.len_bytes,
				},
				ByteVec {
					data: nonce_r.data.clone(),
					len_bytes: nonce_r.len_bytes,
				},
			],
			Some(nonce_joined_le.len()),
		);
		nonce_concat_b.assert_eq("len", nonce_concat.len_bytes, nonce_preimage_len_bytes);
		assert_sha256_message_eq_concat(&nonce_concat_b, &nonce_concat, &nonce_joined_le);

		let nonce_le = nonce_sha256.digest_to_le_wires(b);

		// Base64 requires 48 bytes (6 wires) for alignment, so add zero padding
		let zero = b.add_constant(Word::ZERO);
		let nonce_le_for_base64: Vec<Wire> = nonce_le.into_iter().chain([zero, zero]).collect();

		// The zklogin nonce claim is Base64 URL encoded without padding (i.e.
		// in the same way as JWS components)
		// <https://github.com/MystenLabs/ts-sdks/blob/eb23fc1c122a1495e52d0bd613bf5e8e6eb816cc/packages/typescript/src/zklogin/nonce.ts#L33>
		//
		// The nonce is 32 bytes which encodes to 43 base64 characters.
		// minimal wires those will fit into: 6 wires.
		let base64_check_nonce_builder = b.subcircuit("base64_check_nonce");
		base64_url_safe(
			&base64_check_nonce_builder,
			&nonce_le_for_base64,
			&base64_jwt_payload_nonce,
			base64_check_nonce_builder.add_constant_64(32),
		);

		// Check signing payload. The JWT signed payload L is a concatenation of:
		//
		// L = concat(jwt.header | "." | jwt.payload)
		//
		let max_len_jwt_signing_payload =
			config.max_len_base64_jwt_header() + 1 + config.max_len_base64_jwt_payload();

		// Create witness wires for the JWT signing payload in SHA256 format
		let jwt_signing_payload_sha256_len = b.add_witness();
		let n_words_jwt_signing_payload_sha256 = max_len_jwt_signing_payload;
		let jwt_signing_payload_sha256_message: Vec<Wire> = (0..n_words_jwt_signing_payload_sha256)
			.map(|_| b.add_witness())
			.collect();

		let jwt_signing_payload = ByteVec::new(
			jwt_signing_payload_sha256_message.clone(),
			jwt_signing_payload_sha256_len,
		);

		let jwt_signature_verify =
			Rs256Verify::new(b, jwt_signing_payload, jwt_signature.clone(), rsa_modulus);

		let jwt_signing_payload_le_wires = jwt_signature_verify.sha256.message_to_le_wires(b);
		let signing_joined_words = max_len_jwt_signing_payload;
		let signing_joined_le = jwt_signing_payload_le_wires[..signing_joined_words].to_vec();
		let signing_concat_b = b.subcircuit("jwt_signing_payload_concat");
		let signing_concat = concat(
			&signing_concat_b,
			&[
				ByteVec {
					data: base64_jwt_header.data.clone(),
					len_bytes: base64_jwt_header.len_bytes,
				},
				ByteVec {
					data: vec![signing_concat_b.add_constant_zx_8(b'.')],
					len_bytes: signing_concat_b.add_constant_64(1),
				},
				ByteVec {
					data: base64_jwt_payload.data.clone(),
					len_bytes: base64_jwt_payload.len_bytes,
				},
			],
			Some(signing_joined_le.len()),
		);
		signing_concat_b.assert_eq("len", signing_concat.len_bytes, jwt_signing_payload_sha256_len);
		assert_sha256_message_eq_concat(&signing_concat_b, &signing_concat, &signing_joined_le);

		let jwt_header_attrs = jwt_header_check(b, &jwt_header);
		jwt_payload_check(b, &jwt_payload, &sub, &aud, &iss, &base64_jwt_payload_nonce);

		Self {
			sub,
			aud,
			iss,
			salt,
			zkaddr,
			zkaddr_sha256,
			jwt_header_attrs,
			jwt_signature_verify,
			base64_jwt_header,
			base64_jwt_payload,
			base64_jwt_signature,
			jwt_header,
			jwt_payload,
			jwt_signature,
			base64_jwt_payload_nonce,
			nonce_sha256,
			nonce,
			vk_u,
			t_max,
			nonce_r,
		}
	}

	pub fn populate_sub(&self, w: &mut WitnessFiller, sub_bytes: &[u8]) {
		self.sub.populate_bytes_le(w, sub_bytes);
	}

	pub fn populate_aud(&self, w: &mut WitnessFiller, aud_bytes: &[u8]) {
		self.aud.populate_bytes_le(w, aud_bytes);
	}

	pub fn populate_iss(&self, w: &mut WitnessFiller, iss_bytes: &[u8]) {
		self.iss.populate_bytes_le(w, iss_bytes);
	}

	pub fn populate_salt(&self, w: &mut WitnessFiller, salt_bytes: &[u8]) {
		self.salt.populate_bytes_le(w, salt_bytes);
	}

	pub fn populate_zkaddr(&self, w: &mut WitnessFiller, zkaddr_hash: &[u8; 32]) {
		self.zkaddr_sha256.populate_digest(w, *zkaddr_hash);
	}

	pub fn populate_zkaddr_preimage(&self, w: &mut WitnessFiller, zkaddr_preimage: &[u8]) {
		self.zkaddr_sha256
			.populate_len_bytes(w, zkaddr_preimage.len());
		self.zkaddr_sha256.populate_message(w, zkaddr_preimage);
	}

	pub fn populate_jwt_header(&self, w: &mut WitnessFiller, header_bytes: &[u8]) {
		self.jwt_header.populate_bytes_le(w, header_bytes);
	}

	pub fn populate_jwt_payload(&self, w: &mut WitnessFiller, payload_bytes: &[u8]) {
		self.jwt_payload.populate_bytes_le(w, payload_bytes);
	}

	pub fn populate_jwt_signature(&self, w: &mut WitnessFiller, signature_bytes: &[u8]) {
		assert_eq!(signature_bytes.len(), 256, "RSA signature must be 256 bytes");
		self.jwt_signature.populate_bytes_le(w, signature_bytes);
	}

	pub fn populate_base64_jwt_header(&self, w: &mut WitnessFiller, bytes: &[u8]) {
		self.base64_jwt_header.populate_bytes_le(w, bytes);
	}

	pub fn populate_base64_jwt_payload(&self, w: &mut WitnessFiller, bytes: &[u8]) {
		self.base64_jwt_payload.populate_bytes_le(w, bytes);
	}

	pub fn populate_base64_jwt_signature(&self, w: &mut WitnessFiller, bytes: &[u8]) {
		self.base64_jwt_signature.populate_bytes_le(w, bytes);
	}

	pub fn populate_rsa_modulus(&self, w: &mut WitnessFiller, modulus_bytes: &[u8]) {
		self.jwt_signature_verify
			.modulus
			.populate_bytes_le(w, modulus_bytes);
	}

	pub fn populate_jwt_header_attributes(&self, w: &mut WitnessFiller) {
		// Populate the expected lengths for "alg" and "typ" attributes
		self.jwt_header_attrs[0].populate_len_bytes(w, 5); // "RS256" is 5 bytes
		self.jwt_header_attrs[1].populate_len_bytes(w, 3); // "JWT" is 3 bytes
	}

	pub fn populate_nonce(&self, w: &mut WitnessFiller, nonce_hash: &[u8; 32]) {
		self.nonce_sha256.populate_digest(w, *nonce_hash);
	}

	pub fn populate_nonce_preimage(&self, w: &mut WitnessFiller, nonce_preimage: &[u8]) {
		self.nonce_sha256
			.populate_len_bytes(w, nonce_preimage.len());
		self.nonce_sha256.populate_message(w, nonce_preimage);
	}

	pub fn populate_vk_u(&self, w: &mut WitnessFiller, vk_u_bytes: &[u8; 32]) {
		pack_bytes_into_wires_le(w, &self.vk_u, vk_u_bytes);
	}

	pub fn populate_t_max(&self, w: &mut WitnessFiller, t_max_bytes: &[u8]) {
		self.t_max.populate_bytes_le(w, t_max_bytes);
	}

	pub fn populate_nonce_r(&self, w: &mut WitnessFiller, nonce_r_bytes: &[u8]) {
		self.nonce_r.populate_bytes_le(w, nonce_r_bytes);
	}

	pub fn populate_base64_jwt_payload_nonce(&self, w: &mut WitnessFiller, base64_nonce: &[u8]) {
		// The base64 nonce is 43 characters, but we need to pad to 48 bytes (6 wires)
		let mut padded = vec![0u8; 48];
		padded[..base64_nonce.len()].copy_from_slice(&base64_nonce[..base64_nonce.len()]);
		pack_bytes_into_wires_le(w, &self.base64_jwt_payload_nonce, &padded);
	}
}

/// Asserts that `expected.data[i] == joined_le[i]` for the bytes within `expected.len_bytes`
/// (ignoring trailing bytes past the length).
///
/// This is needed because the SHA-256 message wires past `len_bytes` carry padding bytes
/// (`0x80` delimiter plus the bit-length field), which differ from the zero-padded trailing
/// bytes of the [`concat()`] gadget's output. We mask both sides to the valid byte range and
/// only assert there.
fn assert_sha256_message_eq_concat(
	b: &CircuitBuilder,
	concat_result: &ByteVec,
	joined_le: &[Wire],
) {
	assert_eq!(concat_result.data.len(), joined_le.len());
	for (i, (&a, &e)) in concat_result.data.iter().zip(joined_le).enumerate() {
		let word_byte_offset = i << 3;
		let is_valid_word =
			b.icmp_ult(b.add_constant(Word(word_byte_offset as u64)), concat_result.len_bytes);
		let neg_start = b.add_constant(Word((-(word_byte_offset as i64)) as u64));
		let (bytes_remaining, _) = b.iadd(concat_result.len_bytes, neg_start);
		let mask = create_byte_mask(b, bytes_remaining);
		b.assert_eq_cond(format!("data[{i}]"), b.band(a, mask), b.band(e, mask), is_valid_word);
	}
}

/// A check that verifies that JWT header has the expected constant values in the `alg` and `typ`
/// fields.
///
/// Returns the `(alg, typ)` ByteVecs so callers can populate the runtime length wires.
fn jwt_header_check(b: &CircuitBuilder, jwt_header: &ByteVec) -> [ByteVec; 2] {
	let b = b.subcircuit("jwt_claims_header");
	let alg = ByteVec {
		len_bytes: b.add_inout(),
		data: vec![b.add_constant_64(u64::from_le_bytes(*b"RS256\0\0\0"))],
	};
	let typ = ByteVec {
		len_bytes: b.add_inout(),
		data: vec![b.add_constant_64(u64::from_le_bytes(*b"JWT\0\0\0\0\0"))],
	};
	jwt_claims(&b, jwt_header.len_bytes, &jwt_header.data, &[("alg", &alg), ("typ", &typ)]);
	[alg, typ]
}

/// A check that verifies that the payload has all the claimed values of `sub`, `aud`, `iss`
/// and `nonce`.
fn jwt_payload_check(
	b: &CircuitBuilder,
	jwt_payload: &ByteVec,
	sub_byte_vec: &ByteVec,
	aud_byte_vec: &ByteVec,
	iss_byte_vec: &ByteVec,
	base64_nonce: &[Wire; 6],
) {
	let b = b.subcircuit("jwt_claims_payload");
	// Base64-encoded 32 bytes without padding = 43 chars.
	let nonce_byte_vec = ByteVec {
		len_bytes: b.add_constant_64(43),
		data: base64_nonce.to_vec(),
	};
	jwt_claims(
		&b,
		jwt_payload.len_bytes,
		&jwt_payload.data,
		&[
			("sub", sub_byte_vec),
			("aud", aud_byte_vec),
			("iss", iss_byte_vec),
			("nonce", &nonce_byte_vec),
		],
	);
}

pub struct ZkLoginExample {
	zklogin: ZkLogin,
}

#[derive(Args, Debug, Clone)]
pub struct Params {
	/// Optional config for testing - if None, uses default
	#[clap(skip)]
	pub config: Option<Config>,
}

#[derive(Args, Debug, Clone)]
pub struct Instance {
	/// Subject claim value
	#[arg(long, default_value = "1234567890")]
	pub sub: String,

	/// Audience claim value
	#[arg(long, default_value = "4074087")]
	pub aud: String,

	/// Issuer claim value
	#[arg(long, default_value = "google.com")]
	pub iss: String,

	/// Salt value for zkaddr computation
	#[arg(long, default_value = "test_salt_value")]
	pub salt: String,
}

struct JwtGenerationResult {
	jwt: String,
	zkaddr_hash: [u8; 32],
	vk_u: [u8; 32],
	zkaddr_preimage: Vec<u8>,
	nonce_preimage: Vec<u8>,
	jwt_key_pair: RS256KeyPair,
}

impl JwtGenerationResult {
	fn generate(
		sub: &str,
		aud: &str,
		iss: &str,
		salt: &str,
		rng: &mut impl RngCore,
	) -> Result<Self> {
		// Generate VK_u (verifier public key)
		let mut vk_u = [0u8; 32];
		rng.fill_bytes(&mut vk_u);

		// Fixed values for nonce computation
		let t_max = b"t_max";
		let nonce_r = b"nonce_r";

		// Calculate zkaddr = SHA256(concat(sub, aud, iss, salt))
		let mut zkaddr_preimage = Vec::new();
		zkaddr_preimage.extend_from_slice(sub.as_bytes());
		zkaddr_preimage.extend_from_slice(aud.as_bytes());
		zkaddr_preimage.extend_from_slice(iss.as_bytes());
		zkaddr_preimage.extend_from_slice(salt.as_bytes());
		let zkaddr_hash: [u8; 32] = Sha256::digest(&zkaddr_preimage).into();

		// Calculate nonce = SHA256(concat(vk_u, t_max, nonce_r))
		let mut nonce_preimage = Vec::new();
		nonce_preimage.extend_from_slice(&vk_u);
		nonce_preimage.extend_from_slice(t_max);
		nonce_preimage.extend_from_slice(nonce_r);
		let nonce_hash: [u8; 32] = Sha256::digest(&nonce_preimage).into();
		let nonce_hash_base64 = BASE64_URL_SAFE_NO_PAD.encode(nonce_hash);

		// Generate JWT key pair
		let jwt_key_pair = RS256KeyPair::generate(2048).unwrap();

		// Create and sign JWT
		let claims = Claims::create(Duration::from_hours(2))
			.with_issuer(iss)
			.with_audience(aud)
			.with_subject(sub)
			.with_nonce(nonce_hash_base64);

		let jwt = jwt_key_pair.sign(claims).unwrap();

		Ok(Self {
			jwt,
			zkaddr_hash,
			vk_u,
			zkaddr_preimage,
			nonce_preimage,
			jwt_key_pair,
		})
	}
}

impl ExampleCircuit for ZkLoginExample {
	type Params = Params;
	type Instance = Instance;

	fn build(params: Params, builder: &mut CircuitBuilder) -> Result<Self> {
		let config = params.config.unwrap_or_default();
		let zklogin = ZkLogin::new(builder, config);

		Ok(Self { zklogin })
	}

	fn populate_witness(&self, instance: Instance, w: &mut WitnessFiller) -> Result<()> {
		let mut rng = StdRng::seed_from_u64(42);

		// Generate JWT and related data
		let JwtGenerationResult {
			jwt,
			zkaddr_hash,
			vk_u,
			zkaddr_preimage,
			nonce_preimage,
			jwt_key_pair,
		} = JwtGenerationResult::generate(
			&instance.sub,
			&instance.aud,
			&instance.iss,
			&instance.salt,
			&mut rng,
		)?;

		// Parse JWT components
		let jwt_components = jwt.split(".").collect::<Vec<_>>();
		let [header_base64, payload_base64, signature_base64] = jwt_components.as_slice() else {
			anyhow::bail!("JWT should have format: header.payload.signature");
		};

		// Decode JWT components
		let signature_bytes = BASE64_URL_SAFE_NO_PAD.decode(signature_base64)?;
		let modulus_bytes = jwt_key_pair.public_key().to_components().n;
		let header = BASE64_URL_SAFE_NO_PAD.decode(header_base64)?;
		let payload = BASE64_URL_SAFE_NO_PAD.decode(payload_base64)?;

		ensure!(
			signature_bytes.len() == 256,
			"RSA signature must be 256 bytes, got {}",
			signature_bytes.len()
		);

		// Populate JWT components
		self.zklogin
			.populate_base64_jwt_header(w, header_base64.as_bytes());
		self.zklogin
			.populate_base64_jwt_payload(w, payload_base64.as_bytes());
		self.zklogin
			.populate_base64_jwt_signature(w, signature_base64.as_bytes());
		self.zklogin.populate_jwt_header(w, &header);
		self.zklogin.populate_jwt_header_attributes(w);
		self.zklogin.populate_jwt_payload(w, &payload);
		self.zklogin.populate_jwt_signature(w, &signature_bytes);

		// Populate claim values
		self.zklogin.populate_sub(w, instance.sub.as_bytes());
		self.zklogin.populate_aud(w, instance.aud.as_bytes());
		self.zklogin.populate_iss(w, instance.iss.as_bytes());
		self.zklogin.populate_salt(w, instance.salt.as_bytes());

		// Populate zkaddr
		self.zklogin.populate_zkaddr(w, &zkaddr_hash);
		self.zklogin.populate_zkaddr_preimage(w, &zkaddr_preimage);
		self.zklogin.populate_vk_u(w, &vk_u);
		self.zklogin.populate_t_max(w, b"t_max");
		self.zklogin.populate_nonce_r(w, b"nonce_r");

		// Populate nonce
		let nonce_hash: [u8; 32] = Sha256::digest(&nonce_preimage).into();
		let nonce_hash_base64 = BASE64_URL_SAFE_NO_PAD.encode(nonce_hash);
		self.zklogin.populate_nonce(w, &nonce_hash);
		self.zklogin.populate_nonce_preimage(w, &nonce_preimage);
		self.zklogin
			.populate_base64_jwt_payload_nonce(w, nonce_hash_base64.as_bytes());

		// Populate JWS signature verification data
		let message_str = format!("{header_base64}.{payload_base64}");
		let message = message_str.as_bytes();
		let hash = Sha256::digest(message);
		self.zklogin.populate_rsa_modulus(w, &modulus_bytes);
		self.zklogin
			.jwt_signature_verify
			.populate_len_bytes(w, message.len());
		self.zklogin
			.jwt_signature_verify
			.populate_message(w, message);
		self.zklogin
			.jwt_signature_verify
			.sha256
			.populate_digest(w, hash.into());
		self.zklogin.jwt_signature_verify.populate_intermediates(
			w,
			&signature_bytes,
			&modulus_bytes,
		);

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;

	use super::*;

	fn run_zk_login_with_jwt_population(config: Config) {
		let params = Params {
			config: Some(config),
		};
		let instance = Instance {
			sub: "1234567890".to_string(),
			aud: "4074087".to_string(),
			iss: "google.com".to_string(),
			salt: "test_salt_value".to_string(),
		};

		let mut builder = CircuitBuilder::new();
		let zklogin_example = ZkLoginExample::build(params, &mut builder).unwrap();
		let circuit = builder.build();

		let mut w = circuit.new_witness_filler();
		zklogin_example.populate_witness(instance, &mut w).unwrap();

		circuit.populate_wire_witness(&mut w).unwrap();
		let cs = circuit.constraint_system();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	fn test_zk_login_with_jwt_population() {
		run_zk_login_with_jwt_population(Config::default());
	}

	#[test]
	fn test_zk_login_with_jwt_population_weird_lengths() {
		run_zk_login_with_jwt_population(Config {
			max_len_json_jwt_header: 35,
			max_len_json_jwt_payload: 61,
			max_len_jwt_signature: 32,
			max_len_jwt_sub: 8,
			max_len_jwt_aud: 9,
			max_len_jwt_iss: 10,
			max_len_salt: 9,
			max_len_nonce_r: 6,
			max_len_t_max: 7,
		});
	}
}
