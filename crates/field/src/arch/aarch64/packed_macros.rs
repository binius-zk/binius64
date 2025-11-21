// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	(M128, $scalar:path) => {};
}

macro_rules! maybe_impl_tower_constants {
	($scalar:path, $underlier:ty, _) => {};
	($scalar:path, M128, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M128, { M128::from_u128(alphas!(u128, $alpha_idx)) });
	};
}

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, (None)) => {};
	($impl_macro:ident $name:ident, ($strategy:tt)) => {
		$impl_macro!($name @ $crate::arch::$strategy);
	};
}

macro_rules! impl_transformation {
	($name:ident, ($strategy:ident)) => {
		impl_transformation_with_strategy!($name, $crate::arch::$strategy);
	};
}

pub(crate) use impl_strategy;
pub(crate) use impl_transformation;
pub(crate) use maybe_impl_broadcast;
pub(crate) use maybe_impl_tower_constants;
