// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::ThreadPoolBuildError;

/// Builds the global rayon pool on the current thread when `RAYON_NUM_THREADS=1`.
///
/// A one-thread pool with `use_current_thread` buys two things over rayon's default:
///
/// 1. Throughput close to a build with rayon compiled out.
/// 2. Call stacks with no worker frames, which keeps profiles and debugger sessions readable.
///
/// Call this before anything touches the pool — rayon builds the global pool on first use, and
/// refuses to build it twice. That first use is what the returned error reports, so callers that
/// cannot guarantee they run first should treat it as advisory rather than fatal.
///
/// Calling it more than once is harmless: the result is computed once and cached.
///
/// # Returns
///
/// A reference, because [`ThreadPoolBuildError`] is not `Clone`.
pub fn adjust_thread_pool() -> &'static Result<(), ThreadPoolBuildError> {
	cfg_if! {
		if #[cfg(feature = "rayon")] {
			use std::sync::OnceLock;

			static ONCE_GUARD: OnceLock<Result<(), ThreadPoolBuildError>> = OnceLock::new();

			ONCE_GUARD.get_or_init(|| {
				// Read the environment rather than `current_num_threads`: that call would build
				// the global pool, leaving nothing to override.
				match std::env::var("RAYON_NUM_THREADS") {
					Ok(v) if v == "1" => super::ThreadPoolBuilder::new()
						.num_threads(1)
						.use_current_thread()
						.build_global(),
					_ => Ok(()),
				}
			})
		}
		else {
			static RESULT: Result<(), ThreadPoolBuildError> = Ok(());

			&RESULT
		}
	}
}
