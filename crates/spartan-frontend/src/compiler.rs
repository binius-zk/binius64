// Copyright 2025 Irreducible Inc.

use binius_field::Field;

use crate::{
	circuit_builder::{ConstraintBuilder, ConstraintSystemIR},
	constraint_system::{ConstraintSystem, WitnessLayout},
	wire_elimination::{CostModel, run_wire_elimination},
};

pub fn compile<F: Field>(builder: ConstraintBuilder<F>) -> (ConstraintSystem<F>, WitnessLayout<F>) {
	compile_ir(builder.build())
}

/// Compiles an already-built [`ConstraintSystemIR`] (e.g. from a
/// [`GateRecordingConstraintBuilder`](crate::gate::GateRecordingConstraintBuilder), which produces
/// the IR and a gate sequence together) into the final constraint system and witness layout.
pub fn compile_ir<F: Field>(ir: ConstraintSystemIR<F>) -> (ConstraintSystem<F>, WitnessLayout<F>) {
	let ir = run_wire_elimination(CostModel::default(), ir);
	ir.finalize()
}
