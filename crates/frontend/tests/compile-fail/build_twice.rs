use binius_frontend::CircuitBuilder;

fn main() {
	let builder = CircuitBuilder::new();
	let _circuit = builder.build();
	// The first call above moved `builder`.
	// There is no builder left for the second call to type-check against.
	let _circuit_again = builder.build();
}
