# Accumulative Matrix Sweeping (AMS)

This package contains a publication-oriented AMS manuscript and the normative implementation companion for a bounded-residency virtual-tensor runtime.

## Governing invariant

For every finite supported LLM and finite request, AMS must be able to execute the model on a local memory hierarchy even when the model, an individual tensor, activations, KV state, or output do not fit in accelerator memory or host memory, provided that:

1. the complete model representation and bounded spill state fit in an accessible slower tier;
2. at least one backend can reserve the runtime base and the minimum legal working set of one supported primitive;
3. all executed operators have registered semantics and a stream plan or primitive fallback;
4. required storage, transfers, and kernels eventually complete or return an explicit non-capacity failure; and
5. material allocations on the guaranteed path are brokered or reserved during preflight.

The invariant is a capacity guarantee, not a latency guarantee. AMS trades fast-memory residency for I/O, recomputation, and execution time. It does not hide the dense-weight read lower bound or claim immunity to hardware faults, exhausted backing storage, uncooperative third-party allocators, or unsupported operators.

## Artifacts

- `main.pdf`: compiled 101-page manuscript.
- `main.tex`, `sections/`, and `references.bib`: complete AMS LaTeX source.
- `IMPLEMENTATION_SPEC.md`: normative repository and subsystem contract for an implementation agent.
- `ACCEPTANCE_MATRIX.md`: release gates tied directly to the invariant.
- `schemas/manifest.schema.json`: checkpoint/package manifest contract.
- `schemas/plan.schema.json`: immutable execution-plan and resource-proof contract.
- `schemas/config.schema.json`: runtime configuration contract.
- `schemas/trace.schema.json`: trace bundle and event contract.
- `build.sh`: reproducible local paper build and validation entry point.

## Build

Requirements: a TeX Live installation containing `amsart`, `latexmk`, TikZ, `algorithmicx`, `listings`, and BibTeX.

```bash
./build.sh
```

The script compiles `main.pdf`, fails on unresolved citations/references or overfull boxes, validates every JSON Schema as JSON, and prints PDF metadata. It does not download dependencies or execute untrusted checkpoint data.

## Implementation interpretation

The paper is the mathematical and architectural specification. `IMPLEMENTATION_SPEC.md` defines the implementation order, module ownership, stable contracts, and release gates. The JSON Schemas are normative for serialized artifacts. When prose and schema conflict, the stricter safety requirement applies until the discrepancy is resolved through a versioned specification change.

The initial production slice is deliberately narrower than "all models": it supports a declared decoder-family operator set while preserving the universal fallback theorem for every graph admitted as supported. A model is never silently accepted when operator coverage, backing capacity, or the minimum working set cannot be proved during preflight.

## Evaluation status

The package contains no fabricated performance results. The manuscript defines a benchmark protocol, baselines, hardware matrix, metrics, and acceptance thresholds. An implementation must publish measured artifacts, traces, capability fingerprints, and exact configurations before making performance claims.

## Reconstruction status

Production reconstruction is active on an isolated branch. The first executable slice provides
checked byte arithmetic, stable typed errors, immutable package and conversion-journal descriptors,
canonical JSON, atomic resource-vector admission, a bounded synchronous file store, a hardened
safetensors metadata boundary, restart-safe content-addressed range publication, and a scalar streamed
FP32 linear oracle. The experimental storage path also normalizes multi-shard Hugging Face indexes,
publishes schema-valid AMS manifests last, and implements a deterministic grouped ternary reference
codec with crash-recoverable transformed chunks. An explicit mixed policy can retain sensitive tensors
exactly while ternarizing selected tensors in the same journaled, schema-valid package. It is not yet a
GLM inference engine or an OpenAI-compatible server, and the ternary codec is not a default quality
policy. The CPU semantic oracle can multiply directly from grouped ternary storage with bounded
encoded-group, decoded-group, and output-row tiles; it never reconstructs the matrix in full. A
dependency-free Rust native core now implements the same codec and direct linear path using exclusively
caller-owned scratch buffers. The pinned GLM-5.2 config and Hugging Face index also pass an exact,
fail-closed 59,585-name architecture inventory, including the separate MTP layer and every routed
expert tensor. Deterministic scalar oracles now pin GLM normalization, both rotary layouts, DSA causal
top-k/tie behavior, stable activations, and noaux_tc expert routing; model weight shards have not been
downloaded. A batch-one miniature prefill composes those operators through dense and sparse layers,
IndexShare, routed and shared experts, residuals, and logits while proving that an unselected expert is
never fetched. Its in-memory weights are a semantic fixture, not a production residency claim.

Run the current Windows verification gate with:

```powershell
./ci/verify.ps1
```

See [`docs/RECONSTRUCTION.md`](docs/RECONSTRUCTION.md) for source authority, hardware and model
targets, implementation gates, integration boundaries, and current evidence.
