// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	($underlier:ty, $scalar:path) => {};
}

macro_rules! maybe_impl_tower_constants {
	($scalar:path, $underlier:ty, _) => {};
	($scalar:path, M128, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M128, { M128::from_u128(alphas!(u128, $alpha_idx)) });
	};
	($scalar:path, $underlier:ty, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, $underlier, {
			<$underlier>::from_equal_u128s(alphas!(u128, $alpha_idx))
		});
	};
}

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, (None)) => {};
	($impl_macro:ident $name:ident, (if $cond:ident $gfni_strategy:tt else $fallback:tt)) => {
		cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				$impl_macro!($name @ $crate::arch::$gfni_strategy);
			} else {
				$impl_macro!($name @ $crate::arch::$fallback);
			}
		}
	};
	($impl_macro:ident $name:ident, ($strategy:ident)) => {
		$impl_macro!($name @ $crate::arch::$strategy);
	};
}

macro_rules! impl_transformation {
	($name:ident, (if $cond:ident $num:literal else $fallback:tt)) => {
		cfg_if::cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn!($name, $num);
			} else {
				impl_transformation_with_strategy!($name, $crate::arch::$fallback);
			}
		}
	};
	($name:ident, (if $cond:ident $gfni_strategy:tt else $fallback:tt)) => {
		cfg_if::cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				impl_transformation_with_strategy!($name, $crate::arch::$gfni_strategy);
			} else {
				impl_transformation_with_strategy!($name, $crate::arch::$fallback);
			}
		}
	};
	($name:ident, ($strategy:tt)) => {
		impl_transformation_with_strategy!($name, $crate::arch::$strategy);
	};
}

pub(crate) use impl_strategy;
pub(crate) use impl_transformation;
pub(crate) use maybe_impl_broadcast;
pub(crate) use maybe_impl_tower_constants;
