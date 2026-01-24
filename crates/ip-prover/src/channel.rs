// Copyright 2026 The Binius Developers

use binius_field::Field;

pub trait IPProverChannel<F: Field> {
	fn send_one(&mut self, elem: F);

	fn send_many(&mut self, elems: &[F]) {
		for &elem in elems {
			self.send_one(elem);
		}
	}

	fn sample(&mut self) -> F;
}
