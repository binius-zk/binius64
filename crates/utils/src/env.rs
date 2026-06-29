// Copyright 2024-2025 Irreducible Inc.

/// Read boolean flag from the environment variable.
pub fn boolean_env_flag_set(flag: &str) -> bool {
	std::env::var(flag)
		.is_ok_and(|val| ["1", "on", "ON", "true", "TRUE", "yes", "YES"].contains(&val.as_str()))
}
