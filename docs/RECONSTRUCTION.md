# AMS Reconstruction Charter

Status date: 2026-07-22

## Long-term goal

Reconstruct and production-harden AMS so Z.ai GLM-5.2 can be converted into a provenance-preserving,
mixed ternary/low-bit, shard-streamed AMS package and served locally on the target workstation through
OpenAI-compatible Responses and Chat Completions APIs. Froq must consume that server as an ordinary
model provider. The guaranteed path must remain within configured VRAM, RAM, file-descriptor, I/O,
and spill bounds; reject unsupported work during preflight; preserve tool calls, structured output,
SSE, cancellation, and restart semantics; and publish claims only with correctness, resource,
security, provenance, hardware, and Froq task evidence.

This is a capacity objective, not a promise of interactive GLM-5.2 latency on the target hardware.
Measured throughput and quality gates will be set from reproducible bring-up evidence rather than
projected from parameter counts.

## Source authority

| Surface | Authoritative state | Interpretation |
| --- | --- | --- |
| AMS paper repository | `marius-patrik/accumulative-matrix-sweeping` `main` at `992c3fc5872df722eb4fc943a46ab979214ef66a` | Normative manuscript, implementation specification, acceptance matrix, and JSON Schemas. It contains no prior runtime implementation. |
| AMS reconstruction | Local branch `reconstruct/glm52-runtime` from `992c3fc` | The only active implementation tree. No remote branch or PR has been created. |
| PAES public repository | `marius-patrik/PAES` `main` at `68f3bb69df81a1ecdf88cd2a7daec567ab606f27` | Original PAES 1.0.0 implementation. |
| PAES enterprise recovery | Bundle branch `chore/enterprise-ams-integration` at `d235ac54a13828caac6129de30892d3ff4ff53a8` | Recovered protocol/conformance integration. The bundle is preserved as evidence, not treated as AMS runtime source. |
| Missing AMS gitlink | `813d55ad985dc9d17daae08957d0853b569278bd` | Referenced by recovered PAES but absent from available Git objects. It must not be synthesized or misrepresented as recovered history. |
| Froq personal fork | `rebrand/grok-to-froq` at `b6432a592b3872ca727321c75577d9f0f81e0371` | User-owned worktree with a large uncommitted xai-to-froq rename. Read-only until that work is committed or otherwise isolated. |
| Froq upstream | `xai-org/grok-build` `main` at `a5727c5960452e7527a154b25cb5bf00cda0545e` | Model-agnostic harness reference. It is one commit ahead and two commits behind the current fork ancestry at this status date. |
| GLM-4.7-Flash model | `zai-org/GLM-4.7-Flash` at `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | Official config, generation config, tokenizer metadata/template, README, and safetensors index are pinned locally. The config and index SHA-256 digests are `dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb` and `91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791`. No weight shard has been downloaded. |
| GLM-5.2 model | `zai-org/GLM-5.2` at `b4734de4facf877f85769a911abafc5283eab3d9` | Official config, tokenizer metadata/template, license, README, and safetensors index are pinned locally. No weight shard has been downloaded. |
| GLM-4-MoE-Lite reference | Hugging Face Transformers tag `v5.12.0`, peeled commit `e0e7504bca2bfd1b85bb0eedb148f7b250226f06` | Sparse local checkout of the official configuration, generated/modular model, tests, and documentation used to pin MLA, interleaved RoPE, sigmoid/noaux_tc routing, dense/sparse scheduling, and base-model treatment of MTP weights. |
| GLM-MoE-DSA reference | Hugging Face Transformers tag `v5.12.0`, peeled commit `e0e7504bca2bfd1b85bb0eedb148f7b250226f06` | Sparse local checkout of the official configuration, model, tests, and documentation used to derive execution order and compatibility risks. |

Remote refs were fetched with pruning on the status date. No merge, reset, checkout, push, or remote
history mutation was performed.

## Fixed target

- CPU: AMD Ryzen 5 7600X, 6 cores / 12 threads.
- RAM: approximately 15.6 GiB usable.
- GPU: NVIDIA GeForce RTX 3070, 8 GiB VRAM, compute capability 8.6.
- Local backing storage observed at reconstruction start: approximately 522 GiB free.
- Primary model: Z.ai GLM-5.2, preserving its sparse MoE/DSA topology rather than densifying it.
- Bring-up model: GLM-4.7-Flash after synthetic and miniature GLM fixtures.
- Harness boundary: OpenAI-compatible HTTP; Froq receives only a provider/model configuration pointer.

## Precision policy to validate

The initial package policy is deliberately mixed rather than universally ternary:

- routed expert matrices: ternary candidate, with per-group scales;
- shared/dense MLP matrices: ternary, 2-bit, or 3-bit candidates selected by calibration;
- attention projections: 3-bit or 4-bit candidates;
- embeddings and LM head: 3-bit or 4-bit candidates, including tied-weight semantics;
- router/index/norm/scales: FP16 or FP32 according to sensitivity;
- KV cache: INT8 baseline, then explicitly qualified INT4;
- accumulation: FP32 reference and FP16/FP32 production modes;
- binary weights: experimental only until differential quality evidence supports a declared surface.

No quantization candidate becomes the default merely because it fits. A model package records the
exact tensor policy, calibration corpus digest, converter version/configuration digest, source hashes,
and differential evidence.

## Execution gates

1. **Authority and executable contracts.** Preserve the paper and recovery baselines. Implement
   schemas, immutable descriptors, checked arithmetic, canonical serialization, stable errors,
   resource vectors, and deterministic conversion journal semantics.
2. **Bounded storage proof.** Read safetensors and AMS containers by checked ranges. Convert one
   source shard at a time, atomically publish verified chunks, resume from an idempotent journal, and
   prove that source plus target copies never have to coexist in full.
3. **Operator proof.** Establish exact CPU tiled linear/reduction/elementwise oracles, then fixed-arena
   CUDA execution with reduction-axis and output-axis tiling. Account for payload, workspace, pinned
   buffers, allocator granularity, and completion lifetime.
4. **Miniature GLM proof.** Import a deliberately tiny GLM-shaped fixture that exercises compressed
   Q/K/V projections, sparse attention routing, shared plus routed experts, tokenizer conventions,
   tool-call templates, and multi-token prediction behind an explicit feature flag.
5. **GLM-4.7-Flash proof.** Convert and execute the smaller official architecture; produce operator
   coverage, logits/token differential results, memory high-water evidence, cold/warm I/O traces,
   cancellation/restart tests, and deterministic API transcripts.
6. **OpenAI/Froq compatibility.** Serve `/v1/responses` and `/v1/chat/completions` with streaming,
   structured output, tool calls, usage, error mapping, cancellation, and restart semantics. Exercise
   real Froq coding/tool tasks against the local endpoint without modifying the harness core.
7. **GLM-5.2 conversion and inference.** Stream official source shards into the qualified mixed-bit
   format, preserve sparse execution, validate layerwise and end-to-end quality, and demonstrate
   bounded cold-start, prefill, and decode on the target workstation.
8. **Hardening and publication.** Reconcile the recovered PAES protocol after AMS owns a stable ABI;
   add fuzzing, fault injection, security review, SBOM/provenance, hardware qualification, and reviewed
   PRs. No missing commit is recreated and no benchmark is published without its trace bundle.

Each gate is evidence-driven. Later-gate code may be prototyped behind an experimental flag, but it
cannot be advertised as supported until earlier resource and semantic gates pass.

## Current executable evidence

The reconstruction branch currently contains:

- schema-compatible checked uint63 addition, multiplication, products, and byte ranges;
- the complete stable error-code set required by `IMPLEMENTATION_SPEC.md`;
- immutable storage, byte-range, codec, chunk, tensor-layout, tensor, quantization, and conversion
  journal descriptors;
- deterministic canonical JSON for control-plane artifacts;
- componentwise resource vectors and atomic, idempotently released reservations;
- an immutable synchronous file range store with exact reads and bounded hash verification;
- a strict safetensors header normalizer with duplicate-key, dtype, shape, range, complete-coverage,
  metadata, and allocation-limit checks, differential-tested against the official Python writer;
- bounded content-addressed range publication and an atomically replaced conversion journal whose
  completed chunks survive process restart without rereading source bytes;
- a strict Hugging Face shard-index normalizer that rejects unsafe paths and reconciles every indexed
  tensor, shard header, source content hash, and declared total byte count before planning;
- deterministic, schema-valid AMS identity manifests whose content root covers the canonical manifest
  preimage and whose file becomes visible only after every tensor chunk and graph object is reverified;
- experimental `ams.ternary.trit5` v1 encoding with fixed source-order threshold/scale arithmetic,
  five trits per byte, FP32 group scales, canonical tail padding, bounded source-group reads, and a
  pending-record protocol that recovers the post-transform/pre-journal crash window without rereading
  the source tensor;
- an explicit mixed-policy planner that requires exactly one assignment for every source tensor and
  can publish identity and ternary layouts together under one policy hash, conversion journal,
  schema-valid manifest, and atomic package root;
- a scalar source-order FP16/BF16/FP32 identity linear oracle that streams weights and emits one
  output at a time;
- a direct ternary linear oracle that reads and validates one encoded group, decodes one bounded group,
  accumulates a bounded output-row tile, and matches source-order multiplication over the trusted full
  decoder without reconstructing the parameter matrix.
- a dependency-free, `unsafe`-forbidden Rust core with the complete normative error-code spellings,
  checked in-memory and nonsymlink regular-file positional reads, ternary group decode, arena preflight,
  direct ternary linear execution, and FP16/BF16/FP32 identity linear execution using caller-owned
  scratch; native format/check/test/strict-Clippy gates pass on the Windows MSVC toolchain.
- a bounded, duplicate-key-rejecting GLM-MoE-DSA architecture parser and fail-closed tensor inventory
  that distinguishes dense layers, shared experts, every routed expert, full/shared DSA indexers, and
  MTP tensors. The generated 59,585-name inventory exactly matches the pinned official GLM-5.2 index:
  282 shards and 1,506,659,919,872 declared BF16 tensor bytes. The config and index SHA-256 digests are
  `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a` and
  `5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e`.
- a separate fail-closed GLM-4-MoE-Lite architecture and checkpoint boundary. Its generated 9,703-name
  inventory exactly matches the pinned GLM-4.7-Flash index: one dense inference layer, 46 sparse
  inference layers, and a separately marked 212-tensor MTP layer, including its private embedding and
  shared head. The 48-shard provider mapping declares 31,221,488,576 as `total_size`; the pinned name
  set has SHA-256 `23321d795f0b797ab951613b86cf4d02008e4057b446055fcc2b0265b1f3db3d`.
- deterministic scalar GLM control oracles for RMSNorm, indexer LayerNorm, numerically stable SiLU and
  softmax, interleaved MLA RoPE, half-split indexer RoPE, causal DSA top-k with key-index tie breaking,
  and sigmoid/noaux_tc grouped expert routing. Correction bias affects expert choice but not mixture
  weight, matching the pinned reference order.
- allocation-free native implementations of those same GLM control operators with caller-owned
  outputs and exact scratch requirements for causal DSA selection and expert routing. Cross-language
  constants pin normalization and both RoPE layouts; malformed routing capacity is rejected during
  planning rather than surfacing as an internal execution failure.
- a storage-polymorphic native linear boundary and composed GLM gated MLP. The fixture streams a
  ternary gate, FP32 up projection, and BF16 down projection through one reusable scratch set, matches
  a materialized source-order reference, rejects non-finite inputs before storage reads, and accounts
  for a 136-byte logical high-water: 56 bytes of mixed-linear state plus two five-element FP64
  intermediates.
- allocation-free native sparse-MoE composition for one token. Its 200-byte fixture proof streams the
  router, applies deterministic noaux_tc routing, reads only the selected expert and the shared expert,
  rejects insufficient scratch before the first router read, rejects incomplete expert inventories,
  and leaves caller output unchanged when the shared expert fails after routed work has completed.
- range-streamed native sparse causal attention over separate identity K/V objects. Its 104-byte
  fixture reads only selected vectors, never touches a declared future-token range, rejects noncausal
  indices before I/O, accepts the same indices for IndexShare reuse, and preserves caller output on a
  selected-value numeric failure. This is an operator proof, not yet a paged KV-cache implementation.
- a range-streamed native DSA selector that scans causal offloaded index keys while retaining only
  `top_k` scores and indices. The 72-byte fixture never reads its declared future key, rejects short
  scratch before I/O, and differentially matches the context-sized semantic oracle across causal
  prefixes. Managed scratch is independent of context length; required scan I/O is not.
- a deterministic batch-one miniature GLM-MoE-DSA prefill that composes embedding lookup, a dense
  decoder layer with a full DSA indexer, a sparse layer reusing those indices, causal attention, routed
  plus shared experts, residuals, final normalization, and LM-head logits. Its access-denial invariant
  proves that the deliberately unselected expert is never fetched. Stable logits and token argmaxes are
  pinned as regression anchors. MTP remains an explicit `UNSUPPORTED_OP`, never a silent no-op.
- execution of that same miniature graph from a canonical, manifest-last, exact-inventory AMS package.
  All 69 fixture tensors are declared; the selected routed expert's gate/up/down matrices use trit5
  ternary and the rest use identity FP32. Direct package-range execution exactly matches a trusted
  materialized decoder of those ternary payloads. Content objects are hash-verified lazily before first
  use, a mutated embedding object fails with `INTEGRITY_FAILURE`, required-feature drift and partial
  inventories fail closed, and the measured maximum verification/range buffer in the 64-byte-arena test
  is 64 bytes.

The initial automated gate compiles all Python, passes Ruff, validates every repository JSON Schema as
Draft 2020-12, runs 121 Python tests, and runs 27 Rust tests plus `cargo check` and strict Clippy. The unit
streamed-linear cases use a 340-byte weight object with 12-,
20-, and 64-byte declared working sets. The invariant case uses a 66,548-byte weight object with a
28-byte working arena and exact source-order parity, while verifying that the maximum read plus
accumulator remains within that arena. This proves only the Phase 0 reference behavior; Python
allocator overhead is not claimed as a production memory proof.

The ternary implementation is a format and restart-semantics proof, not a model-quality claim. It uses
an explicit 7/10 mean-absolute threshold and the mean magnitude of selected values as each group scale.
No GLM tensor is assigned this encoding until a complete mixed per-tensor policy and calibration
evidence show that the assignment meets the declared quality threshold.

The miniature GLM provider keeps complete tiny tensors in memory solely to serve as a trusted semantic
fixture. The composed runner itself sees only vector, embedding-row, and linear operations through a
narrow weight-access contract. The AMS package implementation of that contract preserves the
no-unselected-expert-read invariant and bounds individual reads, but it is still a Python semantic
oracle: its allocation overhead, filesystem page cache, and full-object verification traffic are not a
production high-water proof. Those claims remain gated on the native implementation and trace broker.

The safetensors boundary follows the official format contract: an eight-byte little-endian header
length, bounded UTF-8 JSON metadata, relative tensor data offsets, duplicate-key rejection, and complete
non-overlapping coverage of the remaining data buffer. See the
[official safetensors format](https://github.com/huggingface/safetensors#format).

## Known decisions and blockers

- The repository has no license file. A license must be selected by the owner before distributing a
  runtime package; model licenses remain separately recorded in package provenance.
- Rust is the selected native control/data-plane language; Python remains the semantic oracle and
  conversion surface. CUDA kernels will bind through a narrow native boundary after the CPU storage and
  operator contracts are stable.
- Native Windows versus WSL2/Linux packaging is not yet fixed. Contracts and tests remain portable;
  the first CUDA toolchain spike will measure both viable paths before committing the release matrix.
- The host has an up-to-date Rust MSVC toolchain and Visual Studio 2022 Build Tools with bundled
  CMake/Ninja, but no installed CUDA Toolkit or `nvcc`. CUDA code is therefore not yet buildable; the
  NVIDIA driver alone is not treated as evidence of a CUDA development environment.
- Froq configuration is intentionally deferred while its 2,811-entry uncommitted rebrand is in
  progress. The target configuration uses a custom provider with `api_backend = "responses"` and a
  local `/v1` base URL once the server exists.
- Quality and throughput thresholds remain unset until GLM-4.7-Flash produces a reproducible baseline.
  Correctness, bounded residency, and protocol semantics are hard gates regardless of speed.
- The pinned GLM-4.7-Flash index declares `total_size = 31,221,488,576`, but the immutable Hub revision
  reports 62,444,175,504 bytes across its 48 safetensors objects. Twice the declaration is
  62,442,977,152 bytes, leaving 1,198,352 bytes for safetensors headers and file overhead. This strongly
  indicates that the provider wrote a BF16 element count where the generic index contract expects tensor
  bytes. AMS records the anomaly but does not waive byte reconciliation: the current catalog builder
  will reject it until downloaded shard headers independently prove every tensor range and an explicit,
  model-scoped normalization is reviewed.
- Transformers 5.12.0 skips several GLM-MoE-DSA equivalence paths because hard DSA top-k can change
  under small numerical or batching differences, and its assisted/static-cache tests are disabled for
  index-mask incompatibilities. AMS must establish its own deterministic index-ranking, cache, and MTP
  evidence instead of treating upstream model loading as an end-to-end correctness oracle.
