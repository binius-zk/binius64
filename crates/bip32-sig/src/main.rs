// Copyright 2026 The Binius Developers
//! Interactive CLI for BIP32 Bitcoin signatures of knowledge.
//!
//! Proves, in zero knowledge, that you control the seed behind a Bitcoin address and signs a
//! message with it — without revealing the seed. A thin, friendly wrapper over the `bip32` circuit
//! in `binius-examples`.

mod address;
mod circuit;
mod proof_file;

use std::{
	io::{self, Write},
	path::{Path, PathBuf},
	time::Instant,
};

use anyhow::{Result, bail};
use bip39::Mnemonic;
use clap::Parser;
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::{
	address::{
		AddressType, address_for, bind_address_to_hash, bip44_path, derive_compressed_pubkey,
		format_path, parse_address, parse_path, scan_for_path,
	},
	circuit::{MAX_DEPTH, cs_cache_path, load_or_create_cs},
	proof_file::ProofFile,
};

/// Number of addresses to scan when recovering a derivation path for an existing wallet.
const SCAN_LIMIT: usize = 1000;

/// The exact phrase a user must type to acknowledge the unaudited-software warning.
const CONFIRMATION: &str = "I confirm this wallet seed protects no value";

#[derive(Parser, Debug)]
#[command(
	name = "bip32-sig",
	about = "Zero-knowledge BIP32 Bitcoin signatures of knowledge (demo)"
)]
struct Args {
	/// Supply your own BIP39 mnemonic and Bitcoin address instead of a random demo wallet.
	#[arg(long)]
	use_existing_address: bool,

	/// Verify a proof file instead of producing one.
	#[arg(long, value_name = "PROOF_FILE")]
	verify: Option<PathBuf>,
}

fn main() -> Result<()> {
	let args = Args::parse();
	match args.verify {
		Some(proof_path) => run_verify(&proof_path),
		None => run_prove(args.use_existing_address),
	}
}

/// Prompt for a line of input on stdout/stdin and return it trimmed.
fn prompt_line(prompt: &str) -> Result<String> {
	print!("{prompt}");
	io::stdout().flush()?;
	let mut line = String::new();
	io::stdin().read_line(&mut line)?;
	Ok(line.trim().to_string())
}

/// Run the one-time setup if the constraint system is not yet cached, printing a notice. Returns
/// the path it checked so callers can reuse it.
fn first_run_notice(cache: &Path) -> bool {
	if cache.exists() {
		false
	} else {
		println!("First run detected: performing a one-time setup (~1 second)…\n");
		true
	}
}

fn run_prove(use_existing: bool) -> Result<()> {
	let cache = cs_cache_path();
	first_run_notice(&cache);
	// Ensure the constraint system is cached so later runs skip the setup notice.
	load_or_create_cs(&cache)?;

	println!("This is a DEMO of a BIP32 Bitcoin signature of knowledge — unaudited, for");
	println!("experimentation only. By default it invents a random throwaway BIP39 seed phrase");
	println!("and a random BIP44 wallet path. To use your own wallet instead, re-run with");
	println!("--use-existing-address.\n");

	let wallet = if use_existing {
		existing_address_wallet()?
	} else {
		random_demo_wallet()?
	};
	println!("\nAddress:  {}", wallet.address);
	println!("Path:     {}", format_path(&wallet.path));

	let message = prompt_line("\nEnter a message to sign: ")?;
	if message.is_empty() {
		bail!("message must not be empty");
	}

	println!("\nGenerating proof…");
	let proof = circuit::prove(&wallet.seed, &wallet.path, message.as_bytes())?;
	let (setup_time, proving_time, verify_time) =
		(proof.setup_time, proof.proving_time, proof.verify_time);

	let digest_hex = hex::encode(Sha256::digest(message.as_bytes()));
	let filename = format!("signature-{}-{}.bin", wallet.address, &digest_hex[..8]);
	let proof_file = ProofFile {
		pubkey_sha256: proof.pubkey_sha256,
		proof: proof.bytes,
	};
	proof_file.write(Path::new(&filename))?;

	println!("\n✓ Proof generated and self-verified.");
	println!("Proof file:  {filename}");
	println!("Setup time:  {setup_time:.2?}");
	println!("Prove time:  {proving_time:.2?}");
	println!("Verify time: {verify_time:.2?}");
	println!("Proof size:  {} KiB", proof_file.proof.len() / 1024);
	println!("\nVerify with:  bip32-sig --verify {filename}");
	Ok(())
}

fn run_verify(proof_path: &Path) -> Result<()> {
	let cache = cs_cache_path();
	first_run_notice(&cache);
	let cs_load_start = Instant::now();
	let cs = load_or_create_cs(&cache)?;
	let cs_load_time = cs_load_start.elapsed();

	let proof_file = ProofFile::read(proof_path)?;
	let address = parse_address(&prompt_line("Enter the Bitcoin address: ")?)?;
	let message = prompt_line("Enter the signed message: ")?;

	// Bind the address to the committed public-key hash (cheap precondition for verification).
	let ty = match bind_address_to_hash(&address, &proof_file.pubkey_sha256) {
		Ok(ty) => ty,
		Err(e) => {
			println!("\n✗ INVALID — {e}");
			std::process::exit(1);
		}
	};

	match circuit::verify(&cs, &proof_file.pubkey_sha256, &proof_file.proof, message.as_bytes()) {
		Ok(timing) => {
			println!("\n✓ VALID — proof verifies for this {} address and message.", ty.label());
			println!("Setup time:  {:.2?}", cs_load_time + timing.setup_time);
			println!("Verify time: {:.2?}", timing.verify_time);
			Ok(())
		}
		Err(e) => {
			println!("\n✗ INVALID — {e}");
			std::process::exit(1);
		}
	}
}

/// A resolved wallet: the seed, derivation path, and address string to prove against.
struct Wallet {
	seed: [u8; 64],
	path: Vec<u32>,
	address: String,
}

/// Invent a random throwaway wallet. The mnemonic is never printed.
fn random_demo_wallet() -> Result<Wallet> {
	let mut entropy = [0u8; 32]; // 32 bytes -> 24-word mnemonic
	rand::rng().fill_bytes(&mut entropy);
	let mnemonic = Mnemonic::from_entropy(&entropy)
		.map_err(|e| anyhow::anyhow!("failed to generate mnemonic: {e}"))?;
	let seed = mnemonic.to_seed("");

	let ty = prompt_address_type()?;
	let index = rand::rng().next_u32() % 20;
	let path = bip44_path(ty, 0, 0, index);
	let pubkey = derive_compressed_pubkey(&seed, &path)?;
	let address = address_for(ty, &pubkey).to_string();
	Ok(Wallet {
		seed,
		path,
		address,
	})
}

/// Prompt the user for their own mnemonic and address, recovering the derivation path.
fn existing_address_wallet() -> Result<Wallet> {
	// Bold warning (ANSI). Entering a seed into unaudited software is dangerous.
	println!(
		"\x1b[1m⚠  WARNING: This is UNAUDITED software. NEVER enter a seed phrase that protects\n\
		 any real value. Doing so risks total, irreversible loss of funds.\x1b[0m\n"
	);
	if prompt_line(&format!("To proceed, type exactly: {CONFIRMATION}\n> "))? != CONFIRMATION {
		bail!("confirmation phrase not entered; aborting");
	}

	let phrase = prompt_line("\nEnter your BIP39 mnemonic: ")?;
	let mnemonic = Mnemonic::parse_normalized(&phrase)
		.map_err(|e| anyhow::anyhow!("invalid BIP39 mnemonic: {e}"))?;
	let seed = mnemonic.to_seed("");

	let address = parse_address(&prompt_line("Enter your Bitcoin address: ")?)?;
	let ty = AddressType::from_address(&address).ok_or_else(|| {
		anyhow::anyhow!(
			"unsupported address type (only P2PKH, P2WPKH and P2SH-P2WPKH are supported)"
		)
	})?;

	println!("\nScanning up to {SCAN_LIMIT} addresses to recover the derivation path…");
	let path = match scan_for_path(&seed, ty, &address, SCAN_LIMIT)? {
		Some(path) => {
			println!("Recovered path: {}", format_path(&path));
			path
		}
		None => {
			println!("Path not found within {SCAN_LIMIT} addresses.");
			let entered = prompt_line("Enter the BIP32 path manually (e.g. 44'/0'/0'/0/0): ")?;
			let path = parse_path(&entered)?;
			if path.len() > MAX_DEPTH {
				bail!("path depth {} exceeds the circuit maximum of {MAX_DEPTH}", path.len());
			}
			let pubkey = derive_compressed_pubkey(&seed, &path)?;
			if address_for(ty, &pubkey) != address {
				bail!("the entered path does not derive the given address");
			}
			path
		}
	};

	Ok(Wallet {
		seed,
		path,
		address: address.to_string(),
	})
}

/// Present the numbered address-type menu and return the choice.
fn prompt_address_type() -> Result<AddressType> {
	println!("Select a Bitcoin address type:");
	for (i, ty) in AddressType::ALL.iter().enumerate() {
		println!("  {}) {}", i + 1, ty.label());
	}
	loop {
		match prompt_line("Choice [1-3]: ")?.as_str() {
			"1" => return Ok(AddressType::ALL[0]),
			"2" => return Ok(AddressType::ALL[1]),
			"3" => return Ok(AddressType::ALL[2]),
			_ => println!("Please enter 1, 2, or 3."),
		}
	}
}
