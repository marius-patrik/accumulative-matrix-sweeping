# AMS Reconstruction Charter

Status date: 2026-07-23

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
| Froq personal fork | `rebrand/grok-to-froq` at `9821dfe2e48c2e48b8c92244b716d1225153b606` | User-owned clean worktree containing the completed rebrand and Windows build fixes. It remains read-only during AMS reconstruction. |
| Froq upstream | `xai-org/grok-build` `main` at `a5727c5960452e7527a154b25cb5bf00cda0545e` | Model-agnostic harness reference. It is one commit ahead and three commits behind the current fork ancestry at this status date. |
| GLM-4.7-Flash model | `zai-org/GLM-4.7-Flash` at `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | Official config, generation config, complete tokenizer triplet, README, and safetensors index are pinned locally. `tokenizer.json` is 20,217,442 bytes with SHA-256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`; tokenizer config and chat-template hashes are `31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d` and `d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375`. The config and index SHA-256 digests are `dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb` and `91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791`. Shard `model-00002-of-00048.safetensors` is pinned locally at 1,270,648,128 bytes with SHA-256 `8c51e2434efe609cbe652014a924e088a5ea97be35ca29cfa893a1a9a90304b1`; no other weight shard has been downloaded. |
| GLM-5.2 model | `zai-org/GLM-5.2` at `b4734de4facf877f85769a911abafc5283eab3d9` | Official config, generation config, tokenizer/template, license, README, and safetensors index are pinned locally and verified. Config, index, tokenizer, and template SHA-256 digests are `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a`, `5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e`, `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`, and `172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679`. The exact 282-shard LFS inventory is locked by `sha256:a7ed6dcbd48c7740d354d723a2e428ae74daf5e269d5da020b05389f40aab512`; no weight shard has been downloaded. |
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
  can publish identity, ternary, and symmetric-INT4 layouts together under one policy hash, conversion
  journal, schema-valid manifest, and atomic package root;
- a scalar source-order FP16/BF16/FP32 identity linear oracle that streams weights and emits one
  output at a time;
- a direct ternary linear oracle that reads and validates one encoded group, decodes one bounded group,
  accumulates a bounded output-row tile, and matches source-order multiplication over the trusted full
  decoder without reconstructing the parameter matrix.
- a direct symmetric-INT4 linear oracle with the same bounded range contract. The native path validates
  low-first signed nibbles in place and multiplies the stored FP32 scale in FP64, so it matches the
  Python v1 decoder without a decoded-group scratch buffer.
- a dependency-free, `unsafe`-forbidden Rust core with the complete normative error-code spellings,
  checked in-memory and nonsymlink regular-file positional reads, ternary and INT4 decode, arena
  preflight, direct low-bit linear execution, and FP16/BF16/FP32 identity linear execution using
  caller-owned scratch; native format/check/test/strict-Clippy gates pass on the Windows MSVC toolchain.
- a bounded, duplicate-key-rejecting GLM-MoE-DSA architecture parser and fail-closed tensor inventory
  that distinguishes dense layers, shared experts, every routed expert, full/shared DSA indexers, and
  MTP tensors. The generated 59,585-name inventory exactly matches the pinned official GLM-5.2 index:
  282 shards and 1,506,659,919,872 declared BF16 tensor bytes. The config and index SHA-256 digests are
  `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a` and
  `5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e`.
- a complete immutable GLM-5.2 source-header audit. It authenticates the pinned revision's 282 LFS
  names, sizes, and SHA-256 identities under one canonical inventory hash, then proves all 59,585
  remote safetensors headers match the official index. It observed 753,329,940,480 elements,
  1,506,659,919,872 tensor bytes, 1,506,667,387,408 source-file bytes, 59,509 BF16 tensors, and 76
  FP32 tensors while reading 7,467,536 prefix/header bytes and zero tensor-payload bytes. The exact
  result is recorded in `docs/evidence/glm52_source_audit.json` with
  `qualifies_precision_policy = false`; expected LFS hashes are now complete, but payload integrity is
  established only as each shard is staged for conversion.
- a separate fail-closed GLM-4-MoE-Lite architecture and checkpoint boundary. Its generated 9,703-name
  inventory exactly matches the pinned GLM-4.7-Flash index: one dense inference layer, 46 sparse
  inference layers, and a separately marked 212-tensor MTP layer, including its private embedding and
  shared head. The 48-shard provider mapping declares 31,221,488,576 as `total_size`; the pinned name
  set has SHA-256 `23321d795f0b797ab951613b86cf4d02008e4057b446055fcc2b0265b1f3db3d`.
  Mixed-package admission derives the exact row-major shape of every role from the normalized
  architecture and rejects shape drift, including same-element-count transposition, before weight I/O.
- a credential-free, bounded HTTPS range reader that accepts only exact `206` responses with identity
  encoding and matching length/range/object-size metadata. Redirect, status, size, and encoding drift
  fail closed; transport retries are explicitly classified and capped. It was live-probed against an
  immutable, previously undownloaded GLM-4.7-Flash shard using only its prefix and header.
- a complete immutable GLM-4.7-Flash header audit across all 48 shards. The strict parser proves all
  9,703 index mappings, contiguous ranges, 62,442,983,168 tensor bytes, 31,221,488,576 tensor elements,
  62,444,175,504 source-file bytes, and 1,192,336 prefix/header bytes. The checkpoint contains 9,656
  BF16 tensors and 47 FP32 64-element expert-correction biases. The provider's declared
  `total_size = 31,221,488,576` is therefore exactly an element count; tensor bytes exceed twice that
  declaration by 6,016 bytes, exactly the extra two bytes for each of the 3,008 FP32 elements. The
  model-scoped catalog policy permits this interpretation only with the exact pinned index hash;
  standard catalogs remain byte-exact. This is structural header evidence, not full payload integrity:
  only shard 2 has also been downloaded and independently SHA-256 verified.
- a restart-safe ephemeral shard lease built on content-addressed atomic range copy. It transfers and
  full-hash-verifies one immutable shard before exposing a local reader, reuses a completed lease
  without remote I/O, leaves a failed-hash object unpublished, and releases only the exact object path
  under a validated cache marker. A real 1,270,648,128-byte GLM-4.7 shard completed stage and guarded
  release with a fixed 4 MiB buffer. Progressive mixed conversion now consumes this lease while keeping
  source residency to one shard; GLM-5.2 now has complete expected official source hashes but still
  requires a qualified precision policy and per-shard payload verification during conversion.
- an explicitly structural header catalog and progressive mixed plan. The catalog validates every
  header, index mapping, total-size interpretation, expected shard hash, and source size without
  reading tensor payloads or claiming those payload hashes have been verified. The progressive plan
  pins that structure, every assignment, codec hash, target ID, and source object in one deterministic
  plan hash. Its policy hash and target IDs exactly match the existing eager planner, removing two
  possible sources of truth before the durable state machine is added.
- a granular progressive conversion journal and executor. One immutable plan marker is accompanied by
  atomic verified-shard and published-tensor records, avoiding a model-wide journal rewrite for every
  tensor. The persisted records are authoritative: a shard source is released only after its full
  expected hash and exact local header are verified and every tensor output record is durable. Fault
  tests cover interruption after the first tensor, failure after all outputs but before release,
  plan/record disagreement, zero-remote-I/O restart, and two shards that never coexist in the source
  cache. Completed outputs are rehashed before stale-cache cleanup. The cache rejects an unexpected
  second lease and must be disjoint from both output and journal state.
- a completion-only promotion boundary from progressive state back into the established verified
  `HuggingFaceCatalog`, `HuggingFaceMixedPlan`, and `ConversionJournal` contracts. Missing plan,
  shard, or tensor records fail instead of manufacturing completion. A differential integration case
  proves that eager and progressive paths yield identical verified catalogs, plans, journals,
  manifests, and manifest-last published packages.
- an opt-in grouped symmetric INT4 v1 semantic codec. Each group stores one FP32 scale and two signed
  low-nibble-first values per byte; values are restricted to `[-7, 7]`, the `-8` code is rejected,
  tail padding must be canonical zero, and rounding is half away from zero against the stored scale.
  FP16, BF16, and FP32 sources produce the same reviewed encoding, source reads are group-bounded, and
  checksum, non-finite, reserved-code, and padding failures are pinned. A content-addressed INT4
  publication transaction now binds its durable record to the exact source, shape, dtype, and codec
  configuration; completed and pending transactions recover without source rereads, while orphaned,
  corrupt, or plan-disagreeing state fails closed. Eager and shard-progressive policies include the
  selected INT4 configuration in their stable identities and publish `ams.codec.int4.symmetric.v1`
  layouts. The package reader validates that declaration and executes matrices directly from bounded
  group records; a miniature GLM pass has full-decode parity while mixing identity, ternary, and INT4
  tensors. The allocation-free native scalar path shares the mixed-linear dispatch and exact scratch
  union, rejects reserved codes/padding, and matches both the single-group and multi-group Python
  fixtures. This remains experimental and opt-in: hardware-optimized execution and quality
  qualification remain required before a GLM precision policy may select it.
- a deterministic, metadata-only GLM-4 mixed-precision candidate builder and a separate fail-closed
  quality gate. The first candidate preserves embeddings, routers, correction biases, and every norm
  vector exactly; assigns the 9,024 routed-expert matrices to grouped trit5 ternary; and assigns 387
  remaining rank-2 compute matrices to grouped symmetric INT4. On the exact pinned GLM-4.7-Flash
  revision with group size 128, 292 tensors remain identity encoded and the estimated tensor payload is
  9,100,218,112 bytes versus 62,442,983,168 source bytes (6.8617x smaller). Its candidate hash is
  `sha256:3c9c4d9985986f40ea1c04860729c93dd900bbd9f5256558cc0e991cf8dcc0ba` and policy hash is
  `sha256:b954d6e6f55551919d6cfc38eb6d7738bfb21e5530144f1ac6e6d7af06df6717`.
  These values were reproduced by range-reading only 1,192,336 prefix/header bytes with
  `ci/audit_glm4_precision.py` and recorded in
  `docs/evidence/glm47_precision_candidate.json`; they are a structural storage estimate, not
  conversion, integrity, memory, speed, or quality evidence. The separate
  `ci/probe_glm4_quantization.py` boundary independently rederived the candidate and policy identities,
  full-hashed exact official shard `model-00002-of-00048.safetensors` (1,270,648,128 bytes), required
  its 206-tensor header to equal the normalized index subset, and then sampled 64 evenly spaced
  128-value groups from each compressed tensor. It examined 1,638,400 values across 12,800 groups
  while reading 3,276,800 sampled source bytes with a 256-byte maximum sample read. The eight sampled
  INT4 tensors reached 0.992227 cosine similarity and 0.125400 normalized RMS error; the 192 routed
  expert ternary tensors reached 0.900355 and 0.435155, with a 0.422609 reconstructed-zero fraction.
  The exact diagnostic is recorded in
  `docs/evidence/glm47_shard2_quantization_probe.json`. It covers one shard and tensor reconstruction
  only; its schema fixes `qualifies_precision_policy` to false. It neither qualifies the blanket
  routed-expert ternary choice nor substitutes for whole-model logits, tasks, latency, or resource
  evidence. Instead it makes less aggressive expert encodings and ternary calibration variants
  mandatory comparison candidates before full conversion. A candidate can become qualified only
  through evidence that
  identifies the exact source/candidate, calibration and evaluation corpora, evaluator, trusted
  baseline, and candidate runtime, and clears every explicitly supplied token-count, task-count, NLL,
  token-agreement, and task-retention threshold. The repository intentionally supplies no default
  quality thresholds before a real baseline run.
- deterministic scalar GLM control oracles for RMSNorm, indexer LayerNorm, numerically stable SiLU and
  softmax, provider-compatible MLA RoPE (interleaved input pairs emitted as half-split rotated
  components), half-split indexer RoPE, causal DSA top-k with key-index tie breaking, and
  sigmoid/noaux_tc grouped expert routing. Correction bias affects expert choice but not mixture
  weight, matching the pinned reference order. A four-dimensional provider differential, the
  two-dimensional permutation coincidence, and odd-dimension rejection pin the MLA layout.
- allocation-free native implementations of those same GLM control operators with caller-owned
  outputs and exact scratch requirements for causal DSA selection and expert routing. Cross-language
  constants pin normalization and both RoPE layouts; malformed routing capacity is rejected during
  planning rather than surfacing as an internal execution failure.
- a storage-polymorphic native linear boundary and composed GLM gated MLP. The fixture streams a
  ternary gate, FP32 up projection, and BF16 down projection through one reusable scratch set, matches
  a materialized source-order reference, rejects non-finite inputs before storage reads, and accounts
  for a 136-byte logical high-water: 56 bytes of mixed-linear state plus two five-element FP64
  intermediates. The same dispatch now admits symmetric INT4, and its union test proves an eight-byte
  encoded group plus one FP64 accumulator and the identity path's local scalar require 24 bytes.
- allocation-free native sparse-MoE composition for one token. Its 200-byte fixture proof streams the
  router, applies deterministic noaux_tc routing, reads only the selected expert and the shared expert,
  rejects insufficient scratch before the first router read, rejects incomplete expert inventories,
  and leaves caller output unchanged when the shared expert fails after routed work has completed.
- range-streamed native sparse causal attention over separate identity K/V objects. Its 104-byte
  fixture reads only selected vectors, never touches a declared future-token range, rejects noncausal
  indices before I/O, accepts the same indices for IndexShare reuse, and preserves caller output on a
  selected-value numeric failure. This is an operator proof, not yet a paged KV-cache implementation.
- range-streamed native full causal attention for the GLM-4-MoE-Lite path. A one-pass online softmax
  retains one decoded K vector, one decoded V vector, and a transactional concatenated-head output, so
  managed scratch is independent of context length while causal scan I/O remains linear in context.
  Its 72-byte two-head fixture never reads the declared future token; exact-arena, pre-I/O range,
  transactional numeric-failure, and two-pass-softmax differential cases are pinned.
- a native mixed-storage GLM-4-MoE-Lite MLA projector composing `q_a`, Q RMSNorm, `q_b`, `kv_a`, KV
  RMSNorm, `kv_b`, provider-compatible interleaved-input RoPE, and per-head Q/K/V assembly. Its
  296-byte miniature fixture matches an independent source-order reference, preflights all six matrix
  and normalization readers before the first read, and commits Q/K/V only after the whole projection
  succeeds. Identity and ternary linear plans now expose their validated minimum reader lengths so
  composed operators can perform that complete preflight.
- a caller-owned, fixed-capacity native K/V cache for the GLM-4 bring-up path. It admits exact BF16 or
  FP32 key/value arenas and one encoded staging row, requires sequential token positions, performs
  BF16 round-to-nearest-even with non-finite overflow rejection, and makes the authoritative next row
  available as `committed prefix + staged row` without publishing it. Commit copies the complete K/V
  pair and advances the sole authoritative prefix length last. Prefix readers never expose future
  bytes. Non-finite, state-disagreement, out-of-order, short-staging, and over-capacity failures
  preserve both storage and the visible prefix. FP16 and low-bit cache codecs remain explicitly
  unsupported until independently qualified.
- a complete native dense GLM-4-MoE-Lite decoder-layer token path composing input RMSNorm, mixed MLA,
  staged full causal attention, output projection and residual, post-attention RMSNorm, dense gated
  MLP, and the final residual. Its exact 568-byte miniature fixture proves success, a numeric failure
  after the current K/V row has been staged, and successful retry at that same token position. A row is
  committed and caller output becomes visible only after the whole layer succeeds; the late-failure
  case preserves both the prior cache prefix and the caller's output sentinel.
- a complete native sparse-MLP GLM-4-MoE-Lite decoder-layer token path with the same transactional MLA,
  full-attention, output-projection, and residual prefix followed by correction-biased noaux_tc routing,
  selected routed experts, and the shared expert. Its exact 704-byte miniature fixture rejects an
  incomplete expert binding before any weight read, proves a selected-expert numeric failure leaves the
  staged K/V row and caller output unpublished, and then retries the same token successfully.
- a batch-one scalar GLM-4-MoE-Lite prefill oracle over abstract weight access. It executes embedding
  lookup, full causal MLA attention, the dense-then-sparse MLP schedule, routed plus shared experts,
  residuals, final normalization, and LM-head logits while refusing MTP. A published miniature package
  mixes identity, ternary, and INT4 tensors, matches independently decoded package weights exactly,
  proves the unselected expert is never fetched, and pins the manifest architecture as the sole parser
  and exact-inventory authority. This is a semantic/package proof, not native whole-model execution.
- a native GLM-4 decoder-stack transaction composing the first dense layer and all subsequent sparse
  layers with one reusable scratch allocation per layer class. It preflights the complete reader,
  cache, and scratch inventory before any weight read. Its two-layer fixture proves an incomplete later
  binding cannot start the dense layer, a late selected-expert failure rolls the already-committed dense
  KV prefix back, caller output remains untouched, and retry advances both layer prefixes together.
- a native one-token GLM-4 causal-LM wrapper composing identity embedding-row access, the transactional
  decoder stack, final RMSNorm, a mixed-storage LM head, and deterministic lowest-index argmax. Its
  three-token-vocabulary fixture pins 144 bytes of non-layer scratch, rejects a short LM-head binding
  before the embedding reader is touched, rolls both layer caches back when a non-finite LM-head value
  fails after decoder execution, and retries successfully. The plan separately admits full model rows
  and tokenizer-mapped IDs; its deliberately highest unmapped logit cannot be selected or accepted as
  input. The wrapper is manually bound; package plan construction and non-greedy sampling remain open.
- an allocation-free native greedy generation session that borrows one immutable prompt and EOS set,
  preflights the prompt plus worst-case generated-input prefix against the fixed KV capacity, and
  exposes ordered prefill, token, and terminal transitions. Non-final prompt positions execute the
  embedding and decoder without reading the LM head; the final prompt position and subsequent decode
  positions select only tokenizer-mapped logits. Before each call, every layer prefix must equal the
  session position. Token-boundary cancellation, injected failure, and a deliberately stale session
  leave model I/O, session counters, and cache prefixes unchanged; a late LM-head numeric failure
  after prefill rolls back the final decoder row and succeeds on retry. EOS is not emitted or cached,
  while a length stop returns its final emitted token. Sub-token cooperative cancellation remains open.
- an exact GLM-4.7 tokenizer admission and execution boundary. It hashes the complete pinned
  tokenizer/config/template triplet before parsing, rejects symlinks and semantic drift, proves the
  contiguous tokenizer range `0..154855`, and identifies model-logit IDs `154856..154879` as unmapped
  and therefore ineligible for decode selection. The optional runtime uses exact `tokenizers` 0.22.2
  and a Transformers-compatible immutable Jinja sandbox with bounded JSON/render/token inputs. Plain,
  tool-schema, and prior-reasoning chat cases match pinned Transformers 5.12.0 exactly in rendered
  bytes and token IDs; all three decode back to their rendered prompt without loss. The independent
  qualification command is
  `python ci/verify_glm4_tokenizer.py <asset-root> --transformers-oracle`.
- an immutable package-to-native GLM-4 binding descriptor. It deterministically orders every admitted
  tensor, deduplicates referenced storage objects, retains exact role/layer/expert/MTP, shape, dtype,
  range, and codec metadata, and rejects low-bit storage for the vectors the current native core can
  only decode as identity. The runtime policy binds context capacity, tokenizer vocabulary, sorted
  unique EOS IDs, linear arena, and independent BF16/FP32 K/V dtypes while deriving exact cache bytes
  per layer and across the decoder stack. Its semantic hash is relocation-stable because local
  absolute paths are excluded, while the executable descriptor still carries those reviewed paths
  and object hashes. Construction performs zero weight payload reads. The Rust core now independently
  accepts the normalized base-model tensor specs, revalidates every role/layer/expert, exact shape,
  codec byte count, object range, architecture, and runtime policy, and constructs the complete
  dense/sparse decoder plus model plan. Its borrow-scoped callback assembles the exact reader topology
  only after registry count and object lengths match, avoiding self-referential ownership; malformed
  inventory, transposition, vector low-bit storage, codec length, and registry drift fail before
  execution. Python now serializes the exact canonical binding identity separately from its
  machine-local path map. The dependency-pinned `ams-runtime inspect` boundary rejects unknown and
  duplicate JSON fields, verifies the identity hash, independently reconstructs the complete base
  plus MTP inventory and all architecture-derived shapes, re-derives codec hashes/byte counts and
  cache limits, rebuilds the native model plan, then full-hashes every nonsymlink object with a
  bounded buffer through the same retained file handle later exposed to range execution. No admitted
  binding becomes observable after a partial failure. The cross-language miniature package verifies
  52 objects containing all 61 tensors (35 executable and 26 separately marked MTP), and deliberate
  object corruption returns `INTEGRITY_FAILURE`; restoring the exact bytes makes a repeated command
  produce identical evidence. The `ams-runtime generate` boundary now creates exact typed cache and
  scratch ownership from the admitted native plan using fallible allocations, retains Rust session
  and cache state as authoritative, and executes a bounded greedy token-ID request through the same
  verified handles. The mixed identity/ternary/INT4 miniature independently matches the Python
  low-bit oracle at output `[7, 1]`; its evidence is 192 cache heap bytes, 2,107 scratch heap bytes,
  2,155 logical scratch bytes, three state-machine steps, and three committed cache positions. The
  terminal length token is emitted but not consumed into cache. Unknown request fields and
  over-capacity requests fail closed, and a valid retry is byte-for-byte deterministic. The boundary
  now also has a persistent `ams-runtime worker` mode that admits and hashes the binding once, retains
  the exact verified handles and native model plan, and serves strict JSON-line frames capped at
  1 MiB. A one-slot command channel and one active request prevent unbounded queueing; numeric request
  IDs own cancellation, indexed token frames are flushed in source order, and exactly one terminal
  completion/error frame is published while the authoritative request slot is still held. EOF and
  explicit shutdown cancel active work. The process proof streams `[7, 1]`, rejects a concurrent
  request, cancels a full-capacity 15-token prefill, then produces the identical valid result in the
  same PID. A strict Python backend now validates the worker binding/context/vocabulary handshake,
  translates text message histories through the admitted GLM chat template, and uses the pinned
  tokenizers 0.22.2 `DecodeStream` with a terminal full-decode consistency check. The model-backed
  miniature converts a three-token prompt into two native tokens, serves the exact decoded text
  through Responses streaming and Chat non-streaming, restarts after forced worker death, drains a
  disconnected stream without poisoning the next request, and publishes exact 3-input/2-output usage.
  Cache/scratch storage is still fresh per request. This first connected slice rejects reasoning,
  tools, structured output, non-greedy sampling, and prompt-cache hints; those capabilities and cache
  reuse remain open.
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
  ternary, two dense-layer attention projections use grouped INT4, and the rest use identity FP32.
  Direct package-range execution exactly matches a trusted materialized decoder of those low-bit
  payloads. Content objects are hash-verified lazily before first use, a mutated embedding object and
  altered INT4 configuration hash fail with `INTEGRITY_FAILURE`, required-feature drift and partial
  inventories fail closed, and the measured maximum verification/range buffer in the 64-byte-arena
  test is 64 bytes.
- a dependency-free experimental OpenAI/Froq protocol boundary. Responses and Chat Completions inputs
  normalize to one typed request containing source-ordered messages, reasoning context, function calls
  and results, tool schemas, structured-output intent, and sampling controls. Strict UTF-8 JSON parsing
  rejects duplicate keys, unknown provider fields, undeclared tool choices, unsupported hosted tools,
  and request bodies above 16 MiB before inference. The localhost adapter defaults to one admitted
  request, exact model names, optional bearer authentication, typed JSON errors and retry headers, and
  disconnect cancellation with exactly-once slot release. The Responses stream emits ordered
  `response.created`, output-item/content/tool deltas, done frames, `response.completed`, and `[DONE]`;
  the Chat path emits the parallel `chat.completion.chunk` contract. Tests exercise actual HTTP, exact
  multiline reconstruction, late backend failure, overload/retry, and an unconsumed cancelled stream.
  A read-only external probe compiled the current Froq fork's real `SamplingClient` and pinned
  `async-openai` decoder, then exercised both endpoints against this adapter. Both paths reconstructed
  the expected text byte-for-byte and accepted terminal usage of 16 tokens. The probe used a
  deterministic injected backend, so it proves the Froq wire boundary independently. A separate
  miniature test now proves model-backed serving through the same normalized Responses/Chat contract;
  no real Froq coding task or production GLM-4.7/GLM-5.2 package has run yet.

The automated gate compiles all Python, passes Ruff, validates every repository JSON Schema as
Draft 2020-12, passes 212 ordinary Python tests plus four post-build native-process cases, and runs
58 core plus eight runtime Rust tests with `cargo check` and strict Clippy. The unit
streamed-linear cases use a 340-byte weight object with 12-,
20-, and 64-byte declared working sets. The invariant case uses a 66,548-byte weight object with a
28-byte working arena and exact source-order parity, while verifying that the maximum read plus
accumulator remains within that arena. This proves only the Phase 0 reference behavior; Python
allocator overhead is not claimed as a production memory proof.

The ternary implementation is a format and restart-semantics proof, not a model-quality claim. It uses
an explicit 7/10 mean-absolute threshold and the mean magnitude of selected values as each group scale.
No production GLM tensor is assigned this encoding until calibration evidence shows that the complete
mixed per-tensor policy meets the declared quality threshold.

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
- Froq configuration remains read-only. Its own sampler fixtures were used to pin the local
  Responses/Chat stream shapes, and its real client/decoder accepts both current fixture-backed
  endpoints. The eventual pointer is a custom provider with `api_backend = "responses"` and a local
  `/v1` base URL. The adapter now reaches a real AMS miniature, but the Froq pointer remains gated on
  a converted and qualified GLM-4.7 package plus reasoning/tool-call support and a real coding task.
- The protocol adapter currently rejects stored/continued Responses, background mode, hosted tools,
  multiple choices, and unsupported sampling fields instead of silently changing their semantics.
  `json_object` output requires strict duplicate-free JSON and object shape. JSON Schema intent is
  normalized across both endpoints, but the serving boundary rejects it unless a backend explicitly
  declares qualified schema enforcement; that implementation remains a gate. Image input follows the
  same fail-closed capability flag and is disabled by default. Disconnect sets the engine cancellation
  token and releases admission; a backend blocked without yielding must still add cooperative polling
  at its own safe points before cancellation latency can be claimed.
- Quality and throughput thresholds remain unset until GLM-4.7-Flash produces a reproducible baseline.
  Correctness, bounded residency, and protocol semantics are hard gates regardless of speed.
- The pinned GLM-4.7-Flash index uses element-count rather than byte-count `total_size` semantics. All
  48 headers now prove this exactly, and the catalog has a reviewed exception that is unusable unless
  the caller supplies the pinned index content hash. This removes the metadata blocker but does not
  waive integrity: a publishable catalog still hashes every complete source shard before conversion.
- Transformers 5.12.0 skips several GLM-MoE-DSA equivalence paths because hard DSA top-k can change
  under small numerical or batching differences, and its assisted/static-cache tests are disabled for
  index-mask incompatibilities. AMS must establish its own deterministic index-ranking, cache, and MTP
  evidence instead of treating upstream model loading as an end-to-end correctness oracle.
