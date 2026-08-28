// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{num::NonZero, sync::OnceLock, thread::available_parallelism};

use super::ThreadPoolBuildError;

/// Builds the global rayon pool, sized to the machine's physical cores.
///
/// Rayon's own default is one worker per logical CPU.
/// On a machine with simultaneous multithreading that puts two workers on every core.
///
/// The prover's hot loops are carry-less-multiply and bandwidth bound.
/// A second worker on the same core competes for those ports and adds no throughput.
///
/// The environment still chooses the width when it says so:
///
/// ```text
///     RAYON_NUM_THREADS=1   one worker, running on the calling thread
///     RAYON_NUM_THREADS=n   left to rayon, which reads the variable itself
///     unset                 one worker per physical core
/// ```
///
/// A single-worker pool on the calling thread keeps worker frames out of stack traces.
///
/// Rayon builds the global pool on first use and refuses to build it twice.
/// So this must run before anything else touches the pool.
/// A later call reports that earlier use as an error, which a caller may treat as advisory.
///
/// The result is computed once and cached, so calling more than once is harmless.
///
/// # Returns
///
/// A reference, because the error type cannot be cloned.
pub fn adjust_thread_pool() -> &'static Result<(), ThreadPoolBuildError> {
	static ONCE_GUARD: OnceLock<Result<(), ThreadPoolBuildError>> = OnceLock::new();

	ONCE_GUARD.get_or_init(|| {
		// Reading the environment avoids asking rayon for its current width.
		// Asking would build the global pool, leaving nothing left to override.
		match std::env::var("RAYON_NUM_THREADS") {
			// One worker on the calling thread, so no worker frames appear in a stack trace.
			Ok(v) if v == "1" => super::ThreadPoolBuilder::new()
				.num_threads(1)
				.use_current_thread()
				.build_global(),
			// Rayon reads this variable itself.
			// Leaving the pool unbuilt is what lets it apply the requested width.
			Ok(_) => Ok(()),
			// Unset: size the pool to the cores that can actually run in parallel.
			// An unreadable topology falls back to rayon's default rather than guessing.
			Err(_) => physical_core_count().map_or(Ok(()), |n| {
				super::ThreadPoolBuilder::new()
					.num_threads(n.get())
					.build_global()
			}),
		}
	})
}

/// The number of physical cores this process may run on.
///
/// A core is one package-and-core pair in the kernel's topology.
/// Distinct pairs collapse the sibling threads that share a core into one.
///
/// Returns nothing when the topology is unreadable, leaving the choice to the caller.
fn physical_core_count() -> Option<NonZero<usize>> {
	let logical = available_parallelism().ok()?;
	let physical = platform_physical_cores()?;
	// A cgroup quota or an affinity mask can hide part of the machine.
	// Clamping keeps the width within what this process is allowed to use.
	Some(physical.min(logical))
}

/// Counts the distinct package-and-core pairs the kernel reports under `/sys`.
///
/// The kernel exposes one topology directory per online logical CPU.
/// Every sibling of a core repeats that core's pair, so the set of pairs is the set of cores.
#[cfg(target_os = "linux")]
fn platform_physical_cores() -> Option<NonZero<usize>> {
	use std::{collections::HashSet, fs, path::Path};

	let mut cores = HashSet::new();
	for entry in fs::read_dir("/sys/devices/system/cpu").ok()? {
		let path = entry.ok()?.path();

		// Only the per-CPU directories carry a topology.
		// Anything else in this directory is unrelated, so skip it instead of failing.
		//
		//     cpu0, cpu1, ... cpu31   -> read
		//     cpufreq, power, online  -> skip
		let is_cpu_dir = path
			.file_name()
			.and_then(|name| name.to_str())
			.is_some_and(|name| {
				name.starts_with("cpu") && name[3..].bytes().all(|b| b.is_ascii_digit())
			});
		if !is_cpu_dir {
			continue;
		}

		// Each identifier is a single decimal number in its own file.
		let read_id = |field: &str| -> Option<u32> {
			let file: &Path = &path.join("topology").join(field);
			fs::read_to_string(file).ok()?.trim().parse().ok()
		};

		// A CPU missing either identifier is offline or unsupported, so skip it.
		if let (Some(package), Some(core)) = (read_id("physical_package_id"), read_id("core_id")) {
			cores.insert((package, core));
		}
	}

	NonZero::new(cores.len())
}

/// Reports no core count on platforms whose topology this does not read.
#[cfg(not(target_os = "linux"))]
fn platform_physical_cores() -> Option<NonZero<usize>> {
	None
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn physical_core_count_is_within_available_parallelism() {
		// Invariant: the count is a thread-pool width, so it can never exceed the parallelism
		// this process is permitted to use.
		let Some(physical) = physical_core_count() else {
			// A platform without a readable topology has nothing to check.
			return;
		};

		// On this host: 16 physical cores against 32 logical CPUs.
		let logical =
			available_parallelism().expect("available parallelism is known on test hosts");
		assert!(physical <= logical);
	}
}
