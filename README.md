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
indexes and provides an exact-206, identity-encoded HTTPS range reader for immutable public shards,
publishes schema-valid AMS manifests last, and implements a deterministic grouped ternary reference
codec with crash-recoverable transformed chunks. An explicit mixed policy can retain sensitive tensors
exactly while ternarizing selected tensors in the same journaled, schema-valid package. It is not yet a
complete GLM inference engine or a model-backed OpenAI-compatible service, and the ternary codec is not
a default quality policy. The CPU semantic oracle can multiply directly from grouped ternary storage with bounded
encoded-group, decoded-group, and output-row tiles; it never reconstructs the matrix in full. A
dependency-free Rust native core now implements direct ternary and symmetric-INT4 linear paths using
exclusively caller-owned scratch buffers, plus allocation-free identity linear execution directly from
FP16, BF16, or FP32 storage. Its INT4 path validates nibbles in place and performs the stored FP32-scale
multiply in FP64, matching the Python v1 semantics without a decoded-group buffer. It also pins
allocation-free RMSNorm, LayerNorm, SiLU, softmax, both GLM
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
The corresponding sparse GLM-4 layer replaces the dense tail with bounded noaux_tc routing, one
selected-expert working set, and the shared expert. It admits the complete reader inventory before the
first weight read and publishes neither the staged K/V row nor caller output when a late expert fails.
A separate scalar GLM-4-MoE-Lite oracle executes embeddings, full causal attention, the dense/sparse
layer schedule, final normalization, and LM head from the same mixed package boundary used by the
GLM-5 fixture. Its architecture field selects one exact parser and inventory; that inventory contains
9,703 names for the pinned GLM-4.7 model. Package admission also derives and checks the exact row-major
shape for every role, so a transposed matrix with the same element count fails before any payload read.
The tiny fixture proves mixed identity/ternary/INT4 execution and leaves MTP explicitly unsupported.
The native decoder-stack transaction preflights the dense layer and every sparse layer before its
first read, reuses one working set per layer class, and treats all per-layer KV prefixes as one commit:
a later expert failure rolls earlier layer prefixes back so the same token can be retried safely.
The first native causal-LM token wrapper adds exact embedding-row access, final RMSNorm, mixed-storage
LM-head execution, and deterministic lowest-index argmax. It preflights the complete manually bound
model before the embedding read, distinguishes full model rows from tokenizer-mapped IDs, masks
unmapped logits from selection, and rolls the decoder stack back if the final norm or head fails.
A bounded native greedy session now owns the immutable prompt, EOS set, output limit, prompt cursor,
pending decode input, and terminal reason. It preflights the complete worst-case KV capacity, skips
the LM head for every non-final prompt token, validates every layer prefix against its position, emits
ordered prefill/token/terminal steps, and leaves both caches and session retryable after cancellation
or model failure. Cancellation is currently observed between tokens; package binding, non-greedy
sampling, and sub-token cooperative polling remain required.
An immutable package-to-native GLM-4 binding descriptor now closes the control-plane half of package
binding. It orders the exact tensor inventory, maps every range to a deduplicated immutable object,
carries dtype/codec/layer/expert/MTP metadata, admits only native-compatible identity vectors, and
hashes package plus runtime policy without hashing machine-local paths. Context, tokenizer, EOS,
BF16/FP32 cache, linear-arena, and exact per-layer/total KV byte limits are explicit. Descriptor
construction reads no tensor payload; a native reader registry that verifies each declared object
hash and constructs the Rust plans is still required before this becomes model-backed serving.
The official GLM-4.7 tokenizer is now a fail-closed optional runtime boundary rather than a
Transformers dependency. It admits only the exact pinned tokenizer/config/template triplet, proves
contiguous IDs `0..154855`, exposes the 24 model-logit slots with no tokenizer mapping, bounds
render/encode/decode inputs, and reproduces the pinned Transformers 5.12.0 sandboxed chat-template
environment. Plain, tool, and reasoning-history prompts match Transformers byte-for-byte and
token-for-token with `tokenizers` 0.22.2. Install this surface with `pip install -e ".[tokenizer]"`;
the native greedy session is not yet connected to this tokenizer or the OpenAI backend.
An experimental dependency-free localhost adapter now normalizes Froq-shaped Responses and Chat
Completions requests into one typed model contract and emits byte-exact text, reasoning, tool-call,
usage, error, and SSE terminal frames. It is deliberately backend-injected: no fixture response is
presented as model inference, and unsupported persistence, hosted-tool, and provider fields fail during
request preflight.
That reader has audited every header in the pinned 48-shard GLM-4.7-Flash checkpoint without
downloading tensor payloads: all 9,703 index mappings are present and contiguous. The provider's
nonstandard `total_size` is proven to count elements, and that interpretation is accepted only when
the exact pinned index hash is supplied; ordinary Hugging Face catalogs still require byte totals.
A verified ephemeral shard lease can now transfer one immutable source object into a
content-addressed cache, reuse it after interruption without remote I/O, and release only its exact
cache slot after downstream publication. This is the bounded-disk primitive for eventual GLM-5.2
shard-at-a-time conversion. A structural header catalog and deterministic progressive mixed plan now
derive the same policy hash and target IDs as eager conversion without reading tensor payloads; the
durable progressive conversion state machine now records one immutable plan marker plus atomic
per-shard and per-tensor records. It resumes mid-shard without remote rereads, verifies completed
outputs before cleanup, and enforces a single source lease in a cache disjoint from package output.
Only complete durable state can be promoted into the established verified catalog, mixed plan, and
conversion journal. A differential fixture proves that progressive and eager conversion produce the
same plan, journal, manifest, and manifest-last package publication.
The first low-bit candidate is now pinned as the opt-in `ams.int4.symmetric` v1 reference codec:
grouped FP32 scales, signed low-nibble-first values restricted to -7 through 7, deterministic
half-away-from-zero rounding, canonical tail padding, bounded source reads, and strict checksum/
numeric validation. Its content-addressed chunk publisher has an exact configuration/source record,
resumes completed and pending transactions without source rereads, rejects orphaned or corrupt state,
and journals only verified output. Eager and shard-progressive mixed plans now identify its exact
configuration, publish schema-valid manifests, and reload it through a bounded Python direct-linear
path; a miniature GLM pass mixes identity, ternary, and INT4 storage with full-decode parity. INT4 is
still opt-in: hardware-optimized execution and quality qualification remain before any GLM precision
policy.
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
