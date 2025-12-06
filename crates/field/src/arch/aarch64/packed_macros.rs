// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	(M128, $scalar:path) => {};
}

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, (None)) => {};
	($impl_macro:ident $name:ident, ($strategy:tt)) => {
		$impl_macro!($name @ $crate::arch::$strategy);
	};
}

pub(crate) use impl_strategy;
pub(crate) use maybe_impl_broadcast;
