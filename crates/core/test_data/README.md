# Test Data for Serialization Compatibility

This directory contains binary reference files used for testing serialization format compatibility.

## Files

- `constraint_system_v7.bin`: Reference binary serialization of a `ConstraintSystem` using serialization version 7.
- `values_data_v1.bin`: Reference binary serialization of a `ValuesData` using serialization version 1.
- `proof_v1.bin`: Reference binary serialization of a `Proof` using serialization version 1.

## Purpose

These binary files serve as regression tests to ensure that:

1. **Backward compatibility**: Future changes to the serialization format don't accidentally break the ability to deserialize existing data.

2. **Version enforcement**: If breaking changes are made to the serialization format, developers are forced to increment the `SERIALIZATION_VERSION` constant, which will cause the compatibility tests to fail until the version is updated.

3. **Format validation**: The tests verify both the structure and content of deserialized data to ensure the format remains consistent.

## Updating Reference Files

If you make intentional breaking changes to the serialization format:

### For ConstraintSystem
1. Increment `ConstraintSystem::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

### For ValuesData
1. Increment `ValuesData::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_values_data_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

### For Proof
1. Increment `Proof::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_proof_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

## Binary Format

The binary format uses little-endian encoding and follows this structure:

### ConstraintSystem Format
1. **Version header** (4 bytes): `u32` serialization version
2. **Constants**: Vector of `Word` values
3. **Value vector shape**: the five section counts `n_const_pad`, `n_inout`, `n_inout_pad`, `n_private`, `n_private_pad`
4. **ZERO constraints**: Vector of `ZeroConstraint` structures
5. **AND constraints**: Vector of `AndConstraint` structures
6. **IMUL constraints**: Vector of `ImulConstraint` structures
7. **BMUL constraints**: Vector of `BmulConstraint` structures

### ValuesData Format
1. **Version header** (4 bytes): `u32` serialization version
2. **Data**: Vector of `Word` values holding one segment of a value vector

### Proof Format
1. **Version header** (4 bytes): `u32` serialization version
2. **Challenger type**: String identifying the challenger (e.g., "HasherChallenger<Sha256>")
3. **Transcript data**: Vector of bytes containing the proof transcript

All data uses the platform-independent `SerializeBytes`/`DeserializeBytes` traits from `binius-utils`.
