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
- a scalar source-order FP32 linear oracle that streams weights and emits one output at a time.
- a direct ternary linear oracle that reads and validates one encoded group, decodes one bounded group,
  accumulates a bounded output-row tile, and matches source-order multiplication over the trusted full
  decoder without reconstructing the parameter matrix.

The initial automated gate compiles all Python, passes Ruff, validates every repository JSON Schema as
Draft 2020-12, and runs 74 tests. The unit streamed-linear cases use a 340-byte weight object with 12-,
20-, and 64-byte declared working sets. The invariant case uses a 66,548-byte weight object with a
28-byte working arena and exact source-order parity, while verifying that the maximum read plus
accumulator remains within that arena. This proves only the Phase 0 reference behavior; Python
allocator overhead is not claimed as a production memory proof.

The ternary implementation is a format and restart-semantics proof, not a model-quality claim. It uses
an explicit 7/10 mean-absolute threshold and the mean magnitude of selected values as each group scale.
No GLM tensor is assigned this encoding until a complete mixed per-tensor policy and calibration
evidence show that the assignment meets the declared quality threshold.

The safetensors boundary follows the official format contract: an eight-byte little-endian header
length, bounded UTF-8 JSON metadata, relative tensor data offsets, duplicate-key rejection, and complete
non-overlapping coverage of the remaining data buffer. See the
[official safetensors format](https://github.com/huggingface/safetensors#format).

## Known decisions and blockers

- The repository has no license file. A license must be selected by the owner before distributing a
  runtime package; model licenses remain separately recorded in package provenance.
- The production native control/data plane language remains open. The reference contracts are Python;
  fixed-arena CPU/CUDA ownership should move to modern C++ or Rust plus CUDA after the storage proof.
- Native Windows versus WSL2/Linux packaging is not yet fixed. Contracts and tests remain portable;
  the first CUDA toolchain spike will measure both viable paths before committing the release matrix.
- Froq configuration is intentionally deferred while its 2,811-entry uncommitted rebrand is in
  progress. The target configuration uses a custom provider with `api_backend = "responses"` and a
  local `/v1` base URL once the server exists.
- Quality and throughput thresholds remain unset until GLM-4.7-Flash produces a reproducible baseline.
  Correctness, bounded residency, and protocol semantics are hard gates regardless of speed.
