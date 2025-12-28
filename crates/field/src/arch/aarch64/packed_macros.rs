// Copyright 2024-2025 Irreducible Inc.

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, (None)) => {};
	($impl_macro:ident $name:ident, ($strategy:tt)) => {
		$impl_macro!($name @ $crate::arch::$strategy);
	};
}

pub(crate) use impl_strategy;
