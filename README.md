![Binius logo](assets/Logo.png "Binius logo")

# Binius64

Binius64 is a zero-knowledge succinct argument system (zk-SNARK), implemented in Rust. Binius64 is capable of proving arbitrary computations, expressed as non-deterministic circuits over 64-bit words.

Binius64 is a successor to the [original Binius protocol](https://github.com/IrreducibleOSS/binius), with a focus on simplicity and CPU performance. The constraint system natively encodes bitwise operations on 64-bit words through *shifted value indices*, which combine value references with shift operations. This design achieves a 64-fold reduction in constraint complexity compared to bit-level approaches while maintaining the efficiency advantages of binary field arithmetic. The protocol targets modern 64-bit CPUs with SIMD instructions, making it practical for real-world applications requiring zero-knowledge proofs.

For further documentation, visit

* The [binius.xyz](https://www.binius.xyz) documentation website.
  * The [Blueprint](https://www.binius.xyz/blueprint) section provides a detailed description of the Binius64 cryptographic protocol.
  * The [Building](https://www.binius.xyz/building) section contains practical guides for how to build and prove applications using Binius64.
* The Rust docs at [docs.binius.xyz](https://docs.binius.xyz).
* Irreducible's post [Announcing Binius64](https://www.irreducible.com/posts/announcing-binius64).

## Dependencies

- [rustup](https://rustup.rs/): We recommend using rustup to install the Rust compiler and Cargo toolchain.

## Usage

### Building

Binius64 implements optimizations for certain target architectures. To enable these, export the environment variable

```bash
export RUSTFLAGS="-C target-cpu=native"
```

When including binius64 as a dependency, it is recommended to add the following lines to your `Cargo.toml` file to have optimizations across crates

```toml
[profile.release]
lto = "thin"
```

### Running Examples

The `prover/examples/` directory contains example circuits, which you can run using the [CLI framework](https://www.binius.xyz/building/getting-started/cli).

For example, to run an example proving a SHA-512 preimage of a fixed-length 65536-byte message:

```bash
$ RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example sha512 prove --message-len 65536
   Finished `release` profile [optimized + debuginfo] target(s) in 0.09s
     Running `target/release/examples/sha512 prove --message-len 65536`
Building circuit [ 2.99s | 100.00% ]

Setup [ 619.81ms | 100.00% ] { log_inv_rate = 1 }

Generating witness [ 14.12ms | 100.00% ]
├── Input population [ 173.34µs | 1.23% ]
└── Circuit evaluation [ 12.60ms | 89.22% ]

prove [ 128.58ms | 100.00% ] { operation = prove, perfetto_category = operation, n_witness_words = 1048576, n_bitand = 1048576, n_intmul = 1 }
...
```

Pass `--max-message-len` instead of `--message-len` (the two are mutually exclusive) to build the
variable-length circuit, whose capacity is fixed at build time but whose message length is a runtime
witness bounded by that capacity:

```bash
$ RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example sha512 prove --max-message-len 131072
```

### Enabling multithreading

Multithreading using [Rayon](https://github.com/rayon-rs/rayon) is available, but it is disabled by default. This is controlled by the `rayon` Cargo feature. To run an example with multithreading enabled, use `--features rayon`.

## Architecture

For repository structure and architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

Developers (both human and artificial intelligence) should read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

The project welcomes first time contributions from developers who want to learn more about Binius64 and make an impact
on the open source cryptography community.

If you are new to the project and don't know where to start, you can look for [open issues labeled
`good first issue`](https://github.com/IrreducibleOSS/binius/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22) or add
test coverage for existing code. Adding unit tests is a great way to learn how to interact with the codebase, make a
meaningful contribution, and maybe even find bugs!

On the other hand, _we do not accept typo fix PRs from first-time contributors_. These are not significant enough to
justify the additional work for maintainers nor any potential benefits, tangible or intangible, one might get from
being listed as a contributor to the repo.

## Authors & Supporters

Binius64 was originally developed by [Irreducible](https://www.irreducible.com), with funding from
[Bain Capital Crypto](https://baincapitalcrypto.com/) and [Paradigm](https://www.paradigm.xyz/).

## Contributing

Developers (both human and artificial intelligence) should read the [CONTRIBUTING.md](CONTRIBUTING.md) file before submitting pull requests.

The project welcomes first time contributions from developers who want to learn more about Binius64 and make an impact
on the open source cryptography community.

If you are new to the project and don't know where to start, you can look for [open issues labeled
`good first issue`](https://github.com/IrreducibleOSS/binius/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22) or add
test coverage for existing code. Adding unit tests is a great way to learn how to interact with the codebase, make a
meaningful contribution, and maybe even find bugs!

On the other hand, _we do not accept typo fix PRs from first-time contributors_. These are not significant enough to
justify the additional work for maintainers nor any potential benefits, tangible or intangible, one might get from
being listed as a contributor to the repo.

## Licensing

```
SPDX-License-Identifier: Apache-2.0 OR MIT
```

This project is dual-licensed under either [Apache-2.0](LICENSE-Apache-2.0.txt) or [MIT](LICENSE-MIT.txt), at your
option. Any contribution intentionally submitted for inclusion in the project shall be dual-licensed under the
Apache-2.0 and MIT licenses, without any additional terms or conditions.

### Apache-2.0 Notice

  Copyright 2025 The Binius Developers
  Copyright 2025 Irreducible, Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

### MIT Notice

  Copyright (c) 2025 The Binius Developers
  Copyright (c) 2025 Irreducible, Inc.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
