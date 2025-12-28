// Copyright 2024-2025 Irreducible Inc.

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

pub(crate) use impl_strategy;
