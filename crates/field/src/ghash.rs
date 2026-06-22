// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Binary field implementation of GF(2^128) with a modulus of X^128 + X^7 + X^2 + X + 1.
//! This is the GHASH field used in AES-GCM.

use std::{
	fmt::{Debug, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
};
use bytemuck::{Pod, TransparentWrapper, Zeroable};

use super::{
	binary_field::{BinaryField, BinaryField1b, binary_field, impl_field_extension},
	extension::ExtensionField,
};
use crate::{
	AESTowerField8b, Field, WideMul,
	arch::{GhashWideMul, M128, invert_b128, packed_ghash_128::PackedBinaryGhash1x128b},
	arithmetic_traits::{InvertOrZero, Square},
	binary_field_arithmetic::{multiple_using_packed, square_using_packed},
	mul_by_binary_field_1b,
	underlier::{U1, WithUnderlier},
};

binary_field!(pub BinaryField128bGhash(M128), M128::from_u128(0x494ef99794d5244f9152df59d87a9186));

// Convenience `u128` conversions. `binary_field!` already provides `From<M128>`/`From<.. for
// M128>`; these let callers keep constructing/inspecting `BinaryField128bGhash` via `u128`. `M128`
// is a distinct type from `u128` on every target, so these never collide with the macro's impls.
impl From<u128> for BinaryField128bGhash {
	fn from(value: u128) -> Self {
		Self(M128::from(value))
	}
}

impl From<BinaryField128bGhash> for u128 {
	fn from(value: BinaryField128bGhash) -> Self {
		value.0.into()
	}
}

// Deferred-reduction widening multiply via the optimal `GhashWideMul` wrapper applied to the
// width-1 `PackedBinaryGhash1x128b` packing. The scalar and the packing share the `M128` underlier,
// so the conversions are zero-cost reinterprets.
//
// This routes the scalar through the packed type's wrapper rather than wrapping the scalar
// directly. It relies on every M128 GHASH packing using a deferring wrapper whose `Output` is a
// concrete `WideGhashProduct` (not the packed type itself): `WideMul` is a supertrait of both
// `Field` and `PackedField`, and `BinaryField128bGhash` is the `Scalar` of
// `PackedBinaryGhash1x128b`, so a `TrivialWideMul` fallback (whose `Output` is the packed type,
// bounded `Add + Mul`) would close a trait-resolution cycle through `Field: WideMul`.
impl WideMul for BinaryField128bGhash {
	type Output = <GhashWideMul<PackedBinaryGhash1x128b> as WideMul>::Output;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let a = PackedBinaryGhash1x128b::from_underlier(a.to_underlier());
		let b = PackedBinaryGhash1x128b::from_underlier(b.to_underlier());
		<GhashWideMul<PackedBinaryGhash1x128b> as WideMul>::wide_mul(
			GhashWideMul::wrap(a),
			GhashWideMul::wrap(b),
		)
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		let reduced = <GhashWideMul<PackedBinaryGhash1x128b> as WideMul>::reduce(wide);
		Self::from_underlier(GhashWideMul::peel(reduced).to_underlier())
	}
}

unsafe impl Pod for BinaryField128bGhash {}

impl BinaryField128bGhash {
	/// Constructs an element from its `u128` value. The underlier is `M128`, but `u128` is the
	/// ergonomic constructor type, so this converts.
	pub const fn new(value: u128) -> Self {
		Self(M128::from_u128(value))
	}

	#[inline]
	pub fn mul_x(self) -> Self {
		// These scalar bit manipulations are simplest over `u128`; the underlier is `M128`.
		let val: u128 = self.to_underlier().into();
		let shifted = val << 1;

		// GHASH irreducible polynomial: x^128 + x^7 + x^2 + x + 1
		// When the high bit is set, we need to XOR with the reduction polynomial 0x87
		// All 1s if the top bit is set, all 0s otherwise
		let mask = (val >> 127).wrapping_neg();
		let result = shifted ^ (0x87 & mask);

		Self::new(result)
	}

	#[inline]
	pub fn mul_inv_x(self) -> Self {
		// These scalar bit manipulations are simplest over `u128`; the underlier is `M128`.
		let val: u128 = self.to_underlier().into();
		let shifted = val >> 1;

		// If low bit was set, we need to add compensation for the remainder
		// When dividing by x with remainder 1, we add x^(-1) = x^127 to the result
		// Since x^128 ≡ x^7 + x^2 + x + 1, we have x^127 ≡ x^6 + x + 1
		// So 0x43 = x^6 + x + 1 (bits 6, 1, 0) and we set bit 127 for the x^127 term
		// All 1s if the bottom bit is set, all 0s otherwise
		let mask = (val & 1).wrapping_neg();
		let result = shifted ^ (((1u128 << 127) | 0x43) & mask);

		Self::new(result)
	}
}

// Cannot use `impl_arithmetic_using_packed!` because `PackedSubfield<Self, Self>` resolves
// via `Underlier = u128`, but on SIMD targets `PackedBinaryGhash1x128b` uses `M128`.
impl Mul<BinaryField128bGhash> for BinaryField128bGhash {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self {
		crate::tracing::trace_multiplication!(BinaryField128bGhash);
		multiple_using_packed::<PackedBinaryGhash1x128b>(self, rhs)
	}
}

impl Square for BinaryField128bGhash {
	#[inline]
	fn square(self) -> Self {
		square_using_packed::<PackedBinaryGhash1x128b>(self)
	}
}

impl InvertOrZero for BinaryField128bGhash {
	#[inline]
	fn invert_or_zero(self) -> Self {
		invert_b128(self)
	}
}

impl_field_extension!(BinaryField1b(U1) < @7 => BinaryField128bGhash(M128));

mul_by_binary_field_1b!(BinaryField128bGhash);

impl SerializeBytes for BinaryField128bGhash {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.0.serialize(write_buf)
	}
}

impl DeserializeBytes for BinaryField128bGhash {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self(DeserializeBytes::deserialize(read_buf)?))
	}
}

impl FixedSizeSerializeBytes for BinaryField128bGhash {
	const BYTE_SIZE: usize = 16;
}

impl From<AESTowerField8b> for BinaryField128bGhash {
	#[inline]
	fn from(value: AESTowerField8b) -> Self {
		// Raw GHASH values as `u128`, converted to the `M128` underlier at the lookup site so the
		// table needs no const `M128` construction.
		const LOOKUP_TABLE: [u128; 256] = [
			0x00000000000000000000000000000000,
			0x00000000000000000000000000000001,
			0x0dcb364640a222fe6b8330483c2e9849,
			0x0dcb364640a222fe6b8330483c2e9848,
			0x3d5bd35c94646a247573da4a5f7710ed,
			0x3d5bd35c94646a247573da4a5f7710ec,
			0x3090e51ad4c648da1ef0ea02635988a4,
			0x3090e51ad4c648da1ef0ea02635988a5,
			0x6d58c4e181f9199f41a12db1f974f3ac,
			0x6d58c4e181f9199f41a12db1f974f3ad,
			0x6093f2a7c15b3b612a221df9c55a6be5,
			0x6093f2a7c15b3b612a221df9c55a6be4,
			0x500317bd159d73bb34d2f7fba603e341,
			0x500317bd159d73bb34d2f7fba603e340,
			0x5dc821fb553f51455f51c7b39a2d7b08,
			0x5dc821fb553f51455f51c7b39a2d7b09,
			0xa72ec17764d7ced55e2f716f4ede412f,
			0xa72ec17764d7ced55e2f716f4ede412e,
			0xaae5f7312475ec2b35ac412772f0d966,
			0xaae5f7312475ec2b35ac412772f0d967,
			0x9a75122bf0b3a4f12b5cab2511a951c2,
			0x9a75122bf0b3a4f12b5cab2511a951c3,
			0x97be246db011860f40df9b6d2d87c98b,
			0x97be246db011860f40df9b6d2d87c98a,
			0xca760596e52ed74a1f8e5cdeb7aab283,
			0xca760596e52ed74a1f8e5cdeb7aab282,
			0xc7bd33d0a58cf5b4740d6c968b842aca,
			0xc7bd33d0a58cf5b4740d6c968b842acb,
			0xf72dd6ca714abd6e6afd8694e8dda26e,
			0xf72dd6ca714abd6e6afd8694e8dda26f,
			0xfae6e08c31e89f90017eb6dcd4f33a27,
			0xfae6e08c31e89f90017eb6dcd4f33a26,
			0x4d52354a3a3d8c865cb10fbabcf00118,
			0x4d52354a3a3d8c865cb10fbabcf00119,
			0x4099030c7a9fae7837323ff280de9951,
			0x4099030c7a9fae7837323ff280de9950,
			0x7009e616ae59e6a229c2d5f0e38711f5,
			0x7009e616ae59e6a229c2d5f0e38711f4,
			0x7dc2d050eefbc45c4241e5b8dfa989bc,
			0x7dc2d050eefbc45c4241e5b8dfa989bd,
			0x200af1abbbc495191d10220b4584f2b4,
			0x200af1abbbc495191d10220b4584f2b5,
			0x2dc1c7edfb66b7e77693124379aa6afd,
			0x2dc1c7edfb66b7e77693124379aa6afc,
			0x1d5122f72fa0ff3d6863f8411af3e259,
			0x1d5122f72fa0ff3d6863f8411af3e258,
			0x109a14b16f02ddc303e0c80926dd7a10,
			0x109a14b16f02ddc303e0c80926dd7a11,
			0xea7cf43d5eea4253029e7ed5f22e4037,
			0xea7cf43d5eea4253029e7ed5f22e4036,
			0xe7b7c27b1e4860ad691d4e9dce00d87e,
			0xe7b7c27b1e4860ad691d4e9dce00d87f,
			0xd7272761ca8e287777eda49fad5950da,
			0xd7272761ca8e287777eda49fad5950db,
			0xdaec11278a2c0a891c6e94d79177c893,
			0xdaec11278a2c0a891c6e94d79177c892,
			0x872430dcdf135bcc433f53640b5ab39b,
			0x872430dcdf135bcc433f53640b5ab39a,
			0x8aef069a9fb1793228bc632c37742bd2,
			0x8aef069a9fb1793228bc632c37742bd3,
			0xba7fe3804b7731e8364c892e542da376,
			0xba7fe3804b7731e8364c892e542da377,
			0xb7b4d5c60bd513165dcfb96668033b3f,
			0xb7b4d5c60bd513165dcfb96668033b3e,
			0x553e92e8bc0ae9a795ed1f57f3632d4d,
			0x553e92e8bc0ae9a795ed1f57f3632d4c,
			0x58f5a4aefca8cb59fe6e2f1fcf4db504,
			0x58f5a4aefca8cb59fe6e2f1fcf4db505,
			0x686541b4286e8383e09ec51dac143da0,
			0x686541b4286e8383e09ec51dac143da1,
			0x65ae77f268cca17d8b1df555903aa5e9,
			0x65ae77f268cca17d8b1df555903aa5e8,
			0x386656093df3f038d44c32e60a17dee1,
			0x386656093df3f038d44c32e60a17dee0,
			0x35ad604f7d51d2c6bfcf02ae363946a8,
			0x35ad604f7d51d2c6bfcf02ae363946a9,
			0x053d8555a9979a1ca13fe8ac5560ce0c,
			0x053d8555a9979a1ca13fe8ac5560ce0d,
			0x08f6b313e935b8e2cabcd8e4694e5645,
			0x08f6b313e935b8e2cabcd8e4694e5644,
			0xf210539fd8dd2772cbc26e38bdbd6c62,
			0xf210539fd8dd2772cbc26e38bdbd6c63,
			0xffdb65d9987f058ca0415e708193f42b,
			0xffdb65d9987f058ca0415e708193f42a,
			0xcf4b80c34cb94d56beb1b472e2ca7c8f,
			0xcf4b80c34cb94d56beb1b472e2ca7c8e,
			0xc280b6850c1b6fa8d532843adee4e4c6,
			0xc280b6850c1b6fa8d532843adee4e4c7,
			0x9f48977e59243eed8a63438944c99fce,
			0x9f48977e59243eed8a63438944c99fcf,
			0x9283a13819861c13e1e073c178e70787,
			0x9283a13819861c13e1e073c178e70786,
			0xa2134422cd4054c9ff1099c31bbe8f23,
			0xa2134422cd4054c9ff1099c31bbe8f22,
			0xafd872648de276379493a98b2790176a,
			0xafd872648de276379493a98b2790176b,
			0x186ca7a286376521c95c10ed4f932c55,
			0x186ca7a286376521c95c10ed4f932c54,
			0x15a791e4c69547dfa2df20a573bdb41c,
			0x15a791e4c69547dfa2df20a573bdb41d,
			0x253774fe12530f05bc2fcaa710e43cb8,
			0x253774fe12530f05bc2fcaa710e43cb9,
			0x28fc42b852f12dfbd7acfaef2ccaa4f1,
			0x28fc42b852f12dfbd7acfaef2ccaa4f0,
			0x7534634307ce7cbe88fd3d5cb6e7dff9,
			0x7534634307ce7cbe88fd3d5cb6e7dff8,
			0x78ff5505476c5e40e37e0d148ac947b0,
			0x78ff5505476c5e40e37e0d148ac947b1,
			0x486fb01f93aa169afd8ee716e990cf14,
			0x486fb01f93aa169afd8ee716e990cf15,
			0x45a48659d3083464960dd75ed5be575d,
			0x45a48659d3083464960dd75ed5be575c,
			0xbf4266d5e2e0abf497736182014d6d7a,
			0xbf4266d5e2e0abf497736182014d6d7b,
			0xb2895093a242890afcf051ca3d63f533,
			0xb2895093a242890afcf051ca3d63f532,
			0x8219b5897684c1d0e200bbc85e3a7d97,
			0x8219b5897684c1d0e200bbc85e3a7d96,
			0x8fd283cf3626e32e89838b806214e5de,
			0x8fd283cf3626e32e89838b806214e5df,
			0xd21aa2346319b26bd6d24c33f8399ed6,
			0xd21aa2346319b26bd6d24c33f8399ed7,
			0xdfd1947223bb9095bd517c7bc417069f,
			0xdfd1947223bb9095bd517c7bc417069e,
			0xef417168f77dd84fa3a19679a74e8e3b,
			0xef417168f77dd84fa3a19679a74e8e3a,
			0xe28a472eb7dffab1c822a6319b601672,
			0xe28a472eb7dffab1c822a6319b601673,
			0x93252331bf042b11512625b1f09fa87e,
			0x93252331bf042b11512625b1f09fa87f,
			0x9eee1577ffa609ef3aa515f9ccb13037,
			0x9eee1577ffa609ef3aa515f9ccb13036,
			0xae7ef06d2b6041352455fffbafe8b893,
			0xae7ef06d2b6041352455fffbafe8b892,
			0xa3b5c62b6bc263cb4fd6cfb393c620da,
			0xa3b5c62b6bc263cb4fd6cfb393c620db,
			0xfe7de7d03efd328e1087080009eb5bd2,
			0xfe7de7d03efd328e1087080009eb5bd3,
			0xf3b6d1967e5f10707b04384835c5c39b,
			0xf3b6d1967e5f10707b04384835c5c39a,
			0xc326348caa9958aa65f4d24a569c4b3f,
			0xc326348caa9958aa65f4d24a569c4b3e,
			0xceed02caea3b7a540e77e2026ab2d376,
			0xceed02caea3b7a540e77e2026ab2d377,
			0x340be246dbd3e5c40f0954debe41e951,
			0x340be246dbd3e5c40f0954debe41e950,
			0x39c0d4009b71c73a648a6496826f7118,
			0x39c0d4009b71c73a648a6496826f7119,
			0x0950311a4fb78fe07a7a8e94e136f9bc,
			0x0950311a4fb78fe07a7a8e94e136f9bd,
			0x049b075c0f15ad1e11f9bedcdd1861f5,
			0x049b075c0f15ad1e11f9bedcdd1861f4,
			0x595326a75a2afc5b4ea8796f47351afd,
			0x595326a75a2afc5b4ea8796f47351afc,
			0x549810e11a88dea5252b49277b1b82b4,
			0x549810e11a88dea5252b49277b1b82b5,
			0x6408f5fbce4e967f3bdba32518420a10,
			0x6408f5fbce4e967f3bdba32518420a11,
			0x69c3c3bd8eecb4815058936d246c9259,
			0x69c3c3bd8eecb4815058936d246c9258,
			0xde77167b8539a7970d972a0b4c6fa966,
			0xde77167b8539a7970d972a0b4c6fa967,
			0xd3bc203dc59b856966141a437041312f,
			0xd3bc203dc59b856966141a437041312e,
			0xe32cc527115dcdb378e4f0411318b98b,
			0xe32cc527115dcdb378e4f0411318b98a,
			0xeee7f36151ffef4d1367c0092f3621c2,
			0xeee7f36151ffef4d1367c0092f3621c3,
			0xb32fd29a04c0be084c3607bab51b5aca,
			0xb32fd29a04c0be084c3607bab51b5acb,
			0xbee4e4dc44629cf627b537f28935c283,
			0xbee4e4dc44629cf627b537f28935c282,
			0x8e7401c690a4d42c3945ddf0ea6c4a27,
			0x8e7401c690a4d42c3945ddf0ea6c4a26,
			0x83bf3780d006f6d252c6edb8d642d26e,
			0x83bf3780d006f6d252c6edb8d642d26f,
			0x7959d70ce1ee694253b85b6402b1e849,
			0x7959d70ce1ee694253b85b6402b1e848,
			0x7492e14aa14c4bbc383b6b2c3e9f7000,
			0x7492e14aa14c4bbc383b6b2c3e9f7001,
			0x44020450758a036626cb812e5dc6f8a4,
			0x44020450758a036626cb812e5dc6f8a5,
			0x49c93216352821984d48b16661e860ed,
			0x49c93216352821984d48b16661e860ec,
			0x140113ed601770dd121976d5fbc51be5,
			0x140113ed601770dd121976d5fbc51be4,
			0x19ca25ab20b55223799a469dc7eb83ac,
			0x19ca25ab20b55223799a469dc7eb83ad,
			0x295ac0b1f4731af9676aac9fa4b20b08,
			0x295ac0b1f4731af9676aac9fa4b20b09,
			0x2491f6f7b4d138070ce99cd7989c9341,
			0x2491f6f7b4d138070ce99cd7989c9340,
			0xc61bb1d9030ec2b6c4cb3ae603fc8533,
			0xc61bb1d9030ec2b6c4cb3ae603fc8532,
			0xcbd0879f43ace048af480aae3fd21d7a,
			0xcbd0879f43ace048af480aae3fd21d7b,
			0xfb406285976aa892b1b8e0ac5c8b95de,
			0xfb406285976aa892b1b8e0ac5c8b95df,
			0xf68b54c3d7c88a6cda3bd0e460a50d97,
			0xf68b54c3d7c88a6cda3bd0e460a50d96,
			0xab43753882f7db29856a1757fa88769f,
			0xab43753882f7db29856a1757fa88769e,
			0xa688437ec255f9d7eee9271fc6a6eed6,
			0xa688437ec255f9d7eee9271fc6a6eed7,
			0x9618a6641693b10df019cd1da5ff6672,
			0x9618a6641693b10df019cd1da5ff6673,
			0x9bd39022563193f39b9afd5599d1fe3b,
			0x9bd39022563193f39b9afd5599d1fe3a,
			0x613570ae67d90c639ae44b894d22c41c,
			0x613570ae67d90c639ae44b894d22c41d,
			0x6cfe46e8277b2e9df1677bc1710c5c55,
			0x6cfe46e8277b2e9df1677bc1710c5c54,
			0x5c6ea3f2f3bd6647ef9791c31255d4f1,
			0x5c6ea3f2f3bd6647ef9791c31255d4f0,
			0x51a595b4b31f44b98414a18b2e7b4cb8,
			0x51a595b4b31f44b98414a18b2e7b4cb9,
			0x0c6db44fe62015fcdb456638b45637b0,
			0x0c6db44fe62015fcdb456638b45637b1,
			0x01a68209a6823702b0c656708878aff9,
			0x01a68209a6823702b0c656708878aff8,
			0x3136671372447fd8ae36bc72eb21275d,
			0x3136671372447fd8ae36bc72eb21275c,
			0x3cfd515532e65d26c5b58c3ad70fbf14,
			0x3cfd515532e65d26c5b58c3ad70fbf15,
			0x8b49849339334e30987a355cbf0c842b,
			0x8b49849339334e30987a355cbf0c842a,
			0x8682b2d579916ccef3f9051483221c62,
			0x8682b2d579916ccef3f9051483221c63,
			0xb61257cfad572414ed09ef16e07b94c6,
			0xb61257cfad572414ed09ef16e07b94c7,
			0xbbd96189edf506ea868adf5edc550c8f,
			0xbbd96189edf506ea868adf5edc550c8e,
			0xe6114072b8ca57afd9db18ed46787787,
			0xe6114072b8ca57afd9db18ed46787786,
			0xebda7634f8687551b25828a57a56efce,
			0xebda7634f8687551b25828a57a56efcf,
			0xdb4a932e2cae3d8baca8c2a7190f676a,
			0xdb4a932e2cae3d8baca8c2a7190f676b,
			0xd681a5686c0c1f75c72bf2ef2521ff23,
			0xd681a5686c0c1f75c72bf2ef2521ff22,
			0x2c6745e45de480e5c6554433f1d2c504,
			0x2c6745e45de480e5c6554433f1d2c505,
			0x21ac73a21d46a21badd6747bcdfc5d4d,
			0x21ac73a21d46a21badd6747bcdfc5d4c,
			0x113c96b8c980eac1b3269e79aea5d5e9,
			0x113c96b8c980eac1b3269e79aea5d5e8,
			0x1cf7a0fe8922c83fd8a5ae31928b4da0,
			0x1cf7a0fe8922c83fd8a5ae31928b4da1,
			0x413f8105dc1d997a87f4698208a636a8,
			0x413f8105dc1d997a87f4698208a636a9,
			0x4cf4b7439cbfbb84ec7759ca3488aee1,
			0x4cf4b7439cbfbb84ec7759ca3488aee0,
			0x7c6452594879f35ef287b3c857d12645,
			0x7c6452594879f35ef287b3c857d12644,
			0x71af641f08dbd1a0990483806bffbe0c,
			0x71af641f08dbd1a0990483806bffbe0d,
		];

		BinaryField128bGhash::new(LOOKUP_TABLE[value.0 as usize])
	}
}

#[cfg(test)]
mod tests {
	use proptest::{prelude::any, proptest};

	use super::*;
	use crate::{WideMul, binary_field::tests::is_binary_field_valid_generator};

	#[test]
	fn test_ghash_mul() {
		let a = BinaryField128bGhash::new(1u128);
		let b = BinaryField128bGhash::new(1u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(1u128));

		let a = BinaryField128bGhash::new(1u128);
		let b = BinaryField128bGhash::new(2u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(2u128));

		let a = BinaryField128bGhash::new(1u128);
		let b = BinaryField128bGhash::new(1297182698762987u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(1297182698762987u128));

		let a = BinaryField128bGhash::new(2u128);
		let b = BinaryField128bGhash::new(2u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(4u128));

		let a = BinaryField128bGhash::new(2u128);
		let b = BinaryField128bGhash::new(3u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(6u128));

		let a = BinaryField128bGhash::new(3u128);
		let b = BinaryField128bGhash::new(3u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::new(5u128));

		let a = BinaryField128bGhash::from(1u128 << 127);
		let b = BinaryField128bGhash::new(2u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::from(0b10000111));

		let a = BinaryField128bGhash::from((1u128 << 127) + 1);
		let b = BinaryField128bGhash::new(2u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::from(0b10000101));

		let a = BinaryField128bGhash::from(3u128 << 126);
		let b = BinaryField128bGhash::new(2u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::from(0b10000111 + (1u128 << 127)));

		let a = BinaryField128bGhash::from(1u128 << 127);
		let b = BinaryField128bGhash::new(4u128);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::from(0b10000111 << 1));

		let a = BinaryField128bGhash::from(1u128 << 127);
		let b = BinaryField128bGhash::from(1u128 << 122);
		let c = a * b;

		assert_eq!(c, BinaryField128bGhash::from((0b00000111 << 121) + 0b10000111));
	}

	#[test]
	fn test_multiplicative_generator() {
		assert!(is_binary_field_valid_generator::<BinaryField128bGhash>());
	}

	#[test]
	fn test_mul_x() {
		let test_cases = [
			0x0,                                    // Zero
			0x1,                                    // One
			0x2,                                    // Two
			0x80000000000000000000000000000000u128, // High bit set
			0x40000000000000000000000000000000u128, // Second highest bit
			0xffffffffffffffffffffffffffffffffu128, // All bits set
			0x87u128,                               // GHASH reduction polynomial
			0x21ac73a21d46a21badd6747bcdfc5d4d,     // Random value
		];

		for &value in &test_cases {
			let field_val = BinaryField128bGhash::from(value);
			let mul_x_result = field_val.mul_x();
			let regular_mul_result = field_val * BinaryField128bGhash::new(2u128);

			assert_eq!(
				mul_x_result, regular_mul_result,
				"mul_x and regular multiplication by 2 differ for value {:#x}",
				value
			);
		}
	}

	#[test]
	fn test_mul_inv_x() {
		let test_cases = [
			0x0,                                    // Zero
			0x1,                                    // One
			0x2,                                    // Two
			0x1u128,                                // Low bit set
			0x3u128,                                // Two lowest bits set
			0xffffffffffffffffffffffffffffffffu128, // All bits set
			0x87u128,                               // GHASH reduction polynomial
			0x21ac73a21d46a21badd6747bcdfc5d4d,     // Random value
		];

		for &value in &test_cases {
			let field_val = BinaryField128bGhash::from(value);
			let mul_inv_x_result = field_val.mul_inv_x();
			// Safety: 2 is a non-zero field element.
			let regular_mul_result =
				field_val * unsafe { BinaryField128bGhash::new(2u128).invert() };

			assert_eq!(
				mul_inv_x_result, regular_mul_result,
				"mul_inv_x and regular multiplication by 2 differ for value {:#x}",
				value
			);
		}
	}

	proptest! {
		#[test]
		fn test_conversion_from_aes_consistency(a in any::<u8>(), b in any::<u8>()) {
			let a_val = AESTowerField8b::new(a);
			let b_val = AESTowerField8b::new(b);
			let converted_a = BinaryField128bGhash::from(a_val);
			let converted_b = BinaryField128bGhash::from(b_val);
			assert_eq!(BinaryField128bGhash::from(a_val * b_val), converted_a * converted_b);
		}

		#[test]
		fn test_wide_mul_correctness(a in any::<u128>(), b in any::<u128>()) {
			let a = BinaryField128bGhash::from(a);
			let b = BinaryField128bGhash::from(b);
			let reduced = BinaryField128bGhash::reduce(BinaryField128bGhash::wide_mul(a, b));
			assert_eq!(reduced, a * b);
		}

		// Exercises the point of the trait: accumulate two unreduced products, reduce once.
		#[test]
		fn test_wide_mul_deferred_accumulation(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let (a1, b1) = (BinaryField128bGhash::from(a1), BinaryField128bGhash::from(b1));
			let (a2, b2) = (BinaryField128bGhash::from(a2), BinaryField128bGhash::from(b2));
			let wide =
				BinaryField128bGhash::wide_mul(a1, b1) + BinaryField128bGhash::wide_mul(a2, b2);
			assert_eq!(BinaryField128bGhash::reduce(wide), a1 * b1 + a2 * b2);
		}
	}
}
