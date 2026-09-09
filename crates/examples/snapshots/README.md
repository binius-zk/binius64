# Example Circuit Snapshots

This directory contains snapshot files for example circuit statistics. These snapshots are used to ensure that circuit changes are intentional and tracked.

## Usage

For any example circuit (e.g., `sha256`, `zklogin`):

- **Check snapshot**: `cargo run -p binius-examples -- <name> check-snapshot [params]`
- **Update snapshot**: `cargo run -p binius-examples -- <name> bless-snapshot [params]`

### Examples

```bash
# Check sha256 circuit snapshot with specific parameters
cargo run -p binius-examples -- sha256 check-snapshot --max-message-len 64

# Update zklogin circuit snapshot with default parameters
cargo run -p binius-examples -- zklogin bless-snapshot

# View circuit statistics without checking snapshot
cargo run -p binius-examples -- sha256 stat --max-message-len 2048
```

## CI Integration

The GitHub Actions CI workflow automatically checks that circuit statistics match the snapshots on every pull request. If the statistics change, the CI will fail and you'll need to update the snapshot using the `bless-snapshot` command above.

## Snapshot Files

Each example circuit has its own snapshot file:
- `<circuit_name>.snap`: Contains the expected output of the circuit's `stat` command including circuit statistics (number of gates, constraints, witness values, etc.)

## Important Notes

- Snapshots are parameter-dependent. Make sure to use the same parameters when checking/updating snapshots.
- When circuit logic changes intentionally, remember to update the corresponding snapshot.
- The snapshot format includes the circuit name and detailed statistics about the constraint system.