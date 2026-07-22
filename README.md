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
FP16/BF16/FP32 linear oracle. The experimental storage path also normalizes multi-shard Hugging Face
indexes,
publishes schema-valid AMS manifests last, and implements a deterministic grouped ternary reference
codec with crash-recoverable transformed chunks. An explicit mixed policy can retain sensitive tensors
exactly while ternarizing selected tensors in the same journaled, schema-valid package. It is not yet a
GLM inference engine or an OpenAI-compatible server, and the ternary codec is not a default quality
policy. The CPU semantic oracle can multiply directly from grouped ternary storage with bounded
encoded-group, decoded-group, and output-row tiles; it never reconstructs the matrix in full. A
dependency-free Rust native core now implements the same codec and direct linear path using exclusively
caller-owned scratch buffers, plus allocation-free identity linear execution directly from FP16,
BF16, or FP32 storage. It also pins allocation-free RMSNorm, LayerNorm, SiLU, softmax, both GLM
rotary layouts, causal DSA top-k, and noaux_tc expert routing with caller-owned outputs and scratch.
The first native composed subgraph executes a mixed ternary/FP32/BF16 gated MLP from range readers
with one reusable, explicitly accounted scratch set and no matrix materialization.
The GLM-4-MoE-Lite native path composes its four mixed-storage MLA projections, two low-rank RMSNorms,
provider-compatible rotary permutation, and transactional per-head Q/K/V assembly after preflighting
all six weight readers.
Native sparse-MoE composition now streams router logits, applies noaux_tc selection, reads only the
selected routed experts plus the shared expert, and commits output only after the whole token succeeds.
Sparse causal attention likewise range-reads only selected offloaded K/V vectors, accepts reused
IndexShare indices, and uses transactional caller-owned output scratch.
The GLM-4-MoE-Lite path also has a native full causal attention primitive whose online softmax scans
all causal K/V ranges with scratch independent of context length and never reads future-token vectors.
A fixed-capacity caller-owned BF16/FP32 K/V cache can expose a staged next row to the current layer
without advancing its visible prefix, then publish it only after the layer succeeds. The first complete
native dense GLM-4 decoder-layer path composes both RMSNorms, MLA, staged causal attention, output
projection, residuals, and gated MLP transactionally; a late failure leaves both cache and caller output
unchanged so the token position can be retried.
The production DSA selector scans offloaded causal index keys while retaining only top-k state, so its
managed scratch is independent of context length even though scan I/O remains proportional to context.
The pinned GLM-5.2 config and Hugging Face index also pass an exact,
fail-closed 59,585-name architecture inventory, including the separate MTP layer and every routed
expert tensor. Deterministic scalar oracles now pin GLM normalization, both rotary layouts, DSA causal
top-k/tie behavior, stable activations, and noaux_tc expert routing; model weight shards have not been
downloaded. The separately pinned GLM-4.7-Flash bring-up model now passes its own exact 9,703-name
GLM-4-MoE-Lite inventory, including 47 inference layers and a distinct MTP layer; its anomalous
provider `total_size` remains fail-closed pending shard-header reconciliation. A batch-one miniature
prefill composes those operators through dense and sparse layers,
IndexShare, routed and shared experts, residuals, and logits while proving that an unselected expert is
never fetched. The same forward pass now runs from a published 69-tensor AMS package with three selected
expert matrices stored as trit5 ternary; bounded range execution exactly matches the trusted full
decoder, while lazy object hashing detects tampering before first use.

Run the current Windows verification gate with:

```powershell
./ci/verify.ps1
```

See [`docs/RECONSTRUCTION.md`](docs/RECONSTRUCTION.md) for source authority, hardware and model
targets, implementation gates, integration boundaries, and current evidence.
