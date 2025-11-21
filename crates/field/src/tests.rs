// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use proptest::prelude::*;

use crate::{
	AESTowerField8b, BinaryField1b, BinaryField128bGhash, Field, PackedAESBinaryField4x8b,
	PackedAESBinaryField8x8b, PackedAESBinaryField16x8b, PackedAESBinaryField32x8b,
	PackedAESBinaryField64x8b, PackedBinaryField64x1b, PackedBinaryField128x1b,
	PackedBinaryField256x1b, PackedBinaryField512x1b, PackedBinaryGhash1x128b,
	PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
	underlier::{SmallU, WithUnderlier},
};

#[test]
fn test_field_text_debug() {
	assert_eq!(format!("{:?}", BinaryField1b::ONE), "BinaryField1b(0x1)");
	assert_eq!(format!("{:?}", AESTowerField8b::new(127)), "AESTowerField8b(0x7f)");
	assert_eq!(
		format!(
			"{:?}",
			PackedBinaryGhash1x128b::broadcast(BinaryField128bGhash::new(
				162259276829213363391578010288127
			))
		),
		"Packed1x128([0x000007ffffffffffffffffffffffffff])"
	);
	assert_eq!(
		format!("{:?}", PackedAESBinaryField4x8b::broadcast(AESTowerField8b::new(123))),
		"Packed4x8([0x7b,0x7b,0x7b,0x7b])"
	)
}

fn basic_spread<P>(packed: P, log_block_len: usize, block_idx: usize) -> P
where
	P: PackedField,
{
	assert!(log_block_len <= P::LOG_WIDTH);

	let block_len = 1 << log_block_len;
	let repeat = 1 << (P::LOG_WIDTH - log_block_len);
	assert!(block_idx < repeat);

	P::from_scalars(
		packed
			.iter()
			.skip(block_idx * block_len)
			.take(block_len)
			.flat_map(|elem| iter::repeat_n(elem, repeat)),
	)
}

macro_rules! generate_spread_tests_small {
    ($($name:ident, $type:ty, $scalar:ty, $underlier:ty, $width: expr);* $(;)?) => {
        proptest! {
            $(
                #[test]
                #[allow(clippy::modulo_one)]
                fn $name(z in any::<[u8; $width]>()) {
                    let indexed_packed_field = <$type>::from_fn(|i| <$scalar>::from_underlier(<$underlier>::new(z[i])));
                    for log_block_len in 0..=<$type>::LOG_WIDTH {
						for block_idx in 0..(1 <<(<$type>::LOG_WIDTH - log_block_len)) {
							assert_eq!(
								basic_spread(indexed_packed_field, log_block_len, block_idx),
								indexed_packed_field.spread(log_block_len, block_idx)
							);
						}
					}
                }
            )*
        }
    };
}

macro_rules! generate_spread_tests {
    ($($name:ident, $type:ty, $scalar:ty, $underlier:ty, $width: expr);* $(;)?) => {
        proptest! {
            $(
                #[test]
                #[allow(clippy::modulo_one)]
                fn $name(z in any::<[$underlier; $width]>()) {
                    let indexed_packed_field = <$type>::from_fn(|i| <$scalar>::from_underlier(z[i].into()));
                    for log_block_len in 0..=<$type>::LOG_WIDTH {
						for block_idx in 0..(1 <<(<$type>::LOG_WIDTH - log_block_len)) {
							assert_eq!(
								basic_spread(indexed_packed_field, log_block_len, block_idx),
								indexed_packed_field.spread(log_block_len, block_idx)
							);
						}
					}
				}
            )*
        }
    };
}

generate_spread_tests! {
	// 128-bit configurations
	spread_equals_basic_spread_4x128, PackedBinaryGhash4x128b, BinaryField128bGhash, u128, 4;
	spread_equals_basic_spread_2x128, PackedBinaryGhash2x128b, BinaryField128bGhash, u128, 2;
	spread_equals_basic_spread_1x128, PackedBinaryGhash1x128b, BinaryField128bGhash, u128, 1;

	// 8-bit configurations
	spread_equals_basic_spread_64x8, PackedAESBinaryField64x8b, AESTowerField8b, u8, 64;
	spread_equals_basic_spread_32x8, PackedAESBinaryField32x8b, AESTowerField8b, u8, 32;
	spread_equals_basic_spread_16x8, PackedAESBinaryField16x8b, AESTowerField8b, u8, 16;
	spread_equals_basic_spread_8x8, PackedAESBinaryField8x8b, AESTowerField8b, u8, 8;
}

generate_spread_tests_small! {
	// 1-bit configurations
	spread_equals_basic_spread_512x1, PackedBinaryField512x1b, BinaryField1b, SmallU<1>, 512;
	spread_equals_basic_spread_256x1, PackedBinaryField256x1b, BinaryField1b, SmallU<1>, 256;
	spread_equals_basic_spread_128x1, PackedBinaryField128x1b, BinaryField1b, SmallU<1>, 128;
	spread_equals_basic_spread_64x1, PackedBinaryField64x1b, BinaryField1b, SmallU<1>, 64;
}
