# AMS Enterprise Implementation Specification

Status: normative implementation companion to the AMS manuscript  
Version: 1.0-draft  
Date: 2026-07-21

## 1. Normative language and authority

The terms MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, and MAY are normative. The compiled paper defines the mathematical model and proofs. This document defines repository ownership, serialized contracts, lifecycle behavior, release gates, and implementation order. The JSON Schemas in `schemas/` are normative for external serialized artifacts.

A change that weakens the bounded-residency invariant requires a new major specification version. A performance optimization may replace a reference algorithm only when differential, resource, cancellation, and fault tests establish that it preserves the same semantic and resource contracts.

## 2. Product invariant and exact claim boundary

### 2.1 Governing invariant

For a finite supported model `M`, finite request envelope `R`, local hierarchy `H`, and execution policy `P`, successful preflight MUST produce an immutable plan and reservation such that execution never requires the complete model, complete layer, complete activation, complete output, or complete KV cache to reside in the compute-nearest tier or in host DRAM.

The total parameter size may exceed VRAM and DRAM. The runtime MUST trade residency for finite reads, writes, recomputation, and additional execution time. The minimum legal path may operate on one scalar output and one scalar reduction contribution at a time and may use persistent storage as the backing tier.

### 2.2 Preconditions

Preflight may accept only when all of the following are established:

- The model package, tokenizer/configuration assets, request state, journal, and worst-case spill fit within quota-approved backing capacity.
- At least one backend can reserve its fixed runtime base plus one legal primitive working set.
- Every reachable operator has registered semantics and at least one legal lowering under the selected numerical mode.
- Every decoder loop, beam/speculative branch, callback, and custom processor is bounded by the request envelope or rejected.
- Every material allocation in a guaranteed tier is brokered, caller-provided, or included in a measured and reserved fixed backend allowance.
- Codec expansion, alignment padding, kernel workspace, transfer depth, and tail-tile behavior have finite byte bounds.
- Required storage paths and backend operations are available. Availability means that an operation eventually completes or returns a typed error; it does not mean that hardware cannot fail.

### 2.3 Guarantees

The implementation exposes three independent guarantees:

1. **External-memory executability.** A supported finite graph has a finite schedule under the stated assumptions.
2. **Bounded managed allocation.** After accepted preflight, the runtime does not issue a managed allocation beyond the plan reservation.
3. **Operational resilience.** Cooperative reservations, pressure handling, retry, cancellation, and transactional recovery reduce failures caused by shared-system interference. This is not an absolute guarantee against external processes, driver failures, media faults, or kernel defects.

### 2.4 Explicit non-claims

AMS MUST NOT claim that `cudaMemGetInfo`, a fraction of currently free memory, Python object deletion, or an asynchronous-copy flag alone prevents OOM. It MUST NOT silently quantize, prune, approximate attention, reduce the request length, or move to a remote service to make a model fit. It MUST NOT report a model as supported when a reachable operator is opaque or has an unbounded workspace.

## 3. Required top-level user experience

### 3.1 Model opening

`open_model(uri, open_policy)` validates package structure, manifest limits, hashes/signatures according to policy, graph and operator versions, tokenizer assets, and storage reachability. It MUST read metadata and bounded samples only. It MUST NOT materialize the full checkpoint.

### 3.2 Preflight

`model.preflight(envelope, execution_policy)` MUST be mandatory before execution, whether called explicitly or by a convenience method. It returns either:

- an accepted `PreflightResult` containing `plan_id`, `reservation_id`, exact reserved capacities by resource dimension, worst-case spill, selected numerical mode, capability and calibration fingerprints, fallback chain, and estimates; or
- a typed rejection with the shortest actionable cause and evidence.

Preflight MUST attempt smaller legal tiles and declared backend fallbacks before returning `NO_WORKING_SET`. CPU or scalar fallback is required for the universal supported path unless explicitly disabled by policy, in which case the result is a policy rejection rather than a universal-capacity claim.

### 3.3 Session creation and generation

`open_session(preflight_result, generation_request)` verifies that the reservation and model version still match. A session owns request state, task graph epochs, KV mappings, output backpressure, cancellation token, RNG streams, and telemetry correlation. `next_token()` and `stream()` may create or instantiate task templates but MUST NOT introduce a resource class absent from the plan.

### 3.4 Plan visibility

Users and operators MUST be able to inspect a plan without executing it. Inspection includes:

- selected backend and fallback chain;
- operator implementation IDs and tile shapes;
- arenas, fixed reserves, queue credits, and in-flight depth;
- expected physical bytes by storage path;
- estimated latency components and confidence;
- numerical contract and determinism declaration;
- guards and replan conditions;
- unsupported or policy-disabled alternatives;
- proof summary mapping every live buffer and task workspace to a reservation item.

## 4. Repository and process architecture

The production repository SHOULD use the following responsibility boundaries. Exact languages may vary, but dependencies MUST preserve this layering.

```text
ams/
  pyproject.toml
  CMakeLists.txt
  Cargo.toml                    # optional when Rust owns native control plane
  README.md
  LICENSE
  SECURITY.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  CHANGELOG.md
  docs/
  schemas/
  src/ams/
    api.py
    config.py
    errors.py
    version.py
    ir/
    compiler/
    planner/
    runtime/
    storage/
    backends/
    ops/
    inference/
    training/
    distributed/
    telemetry/
    security/
    integrations/
  native/
    include/ams/
    core/
    storage/
    backends/
    kernels/
  tools/
    convert.py
    inspect.py
    preflight.py
    benchmark.py
    trace.py
    recover.py
    validate_package.py
  tests/
    unit/
    property/
    differential/
    invariant/
    fault/
    integration/
    models/
    security/
    performance/
    fixtures/
  benchmarks/
  examples/
  ci/
```

### 4.1 Dependency rules

- `ir` has no dependency on PyTorch, storage implementations, backend implementations, or the runtime.
- `compiler` imports framework graphs and produces canonical IR. It depends only on abstract checkpoint metadata and capability descriptors.
- `planner` consumes canonical IR, shapes/guards, storage layouts, calibrated capabilities, and policy. It produces a serialized plan and proof summary. It does not submit work.
- `runtime` consumes serialized plans and abstract backend/store interfaces. It never receives live PyTorch graph objects.
- `ops` owns semantic definitions, candidate generation, resource formulas, reference implementations, and backend lowering IDs.
- framework integrations are optional packages above the public API and compiler import surfaces.
- native completion paths MUST NOT invoke arbitrary Python while holding broker, scheduler, storage, or device locks.
- public examples and third-party plugins use only public APIs. Architecture tests enforce imports and native link dependencies.

### 4.2 Recommended language split

A defensible split is Python for public API, graph import, high-level planner orchestration, tools, and documentation; Rust or modern C++ for parsers, broker, scheduler, storage data path, event loop, transactional state, and backend ABI; and CUDA/HIP/Metal/C++ kernels for compute. A pure-Python reference executor is required for semantic testing but is not the guaranteed production data path.

All native ownership MUST be RAII-based. All byte arithmetic MUST use checked unsigned operations. Parser and scheduler code MUST compile with warnings as errors and run under applicable sanitizers.

## 5. Core immutable value model

### 5.1 Identifiers

External identifiers are opaque UTF-8 strings with declared maximum lengths. Content identifiers use an algorithm-qualified digest such as `sha256:<hex>`. Runtime-local handles are generation-checked integers or UUIDs and MUST NOT be reused while stale completions may exist.

Required identifiers include:

- package/model root;
- graph and IR version;
- tensor and unique storage object;
- physical layout and codec;
- plan, reservation, session, request, epoch, task, event, and lease;
- transaction, journal entry, page/version, hardware fingerprint, and calibration profile.

### 5.2 Shapes and arithmetic

Shapes, strides, extents, offsets, and byte lengths use non-negative checked integers. Every multiplication/addition used to compute a byte range MUST detect overflow before allocation or I/O. Symbolic dimensions carry finite upper guards for accepted plans. Negative strides, overlapping views, and zero-sized tensors require explicit semantic handling rather than ad hoc pointer arithmetic.

### 5.3 Resource vector

Every task and plan uses a complete resource vector. At minimum:

```text
ResourceVector {
  device_bytes[device_id]
  host_pinned_bytes[node_id]
  host_pageable_bytes[node_id]
  spill_bytes[store_id]
  device_copy_slots[path_id]
  device_compute_slots[queue_id]
  io_requests[store_id]
  decode_workers[codec_id]
  file_descriptors
  journal_transactions[store_id]
  network_bytes_in_flight[path_id]   # distributed/local fabric only
}
```

Vectors are compared componentwise. Admission MUST be atomic across the complete vector. Partial acquisition is forbidden unless the operation is modeled as a distinct task that can make progress with the acquired subset and holds no unrelated resource while waiting.

### 5.4 Numeric contract

Every plan and operator records one of:

- `reference`: scalar or explicitly ordered semantics used as oracle;
- `deterministic`: fixed reduction trees, algorithms, seeds, and backend settings;
- `stable`: numerically stable algorithms with tolerance contracts;
- `fast`: permitted reassociation and vendor kernels with documented nondeterminism;
- `approximate`: explicit quantization/sparsity/lossy policy and quality metadata.

Approximate storage and compute are opt-in. Capacity fallback MUST preserve the requested numerical class or reject; it may not silently downgrade.

## 6. Virtual tensor subsystem

### 6.1 Logical abstraction

A `VirtualTensor` describes logical dtype, shape, strides/views, tensor class, mutability, version, aliases, and one or more physical layouts. It does not imply residency. A `Tile` is a closed-open hyperrectangle `(origin, extent)` validated against logical shape.

Required tensor classes are parameter, constant, activation, accumulator, KV/recurrent state, output, gradient, optimizer state, and temporary metadata. The planner may assign different caching, durability, and eviction policies by class.

### 6.2 Tile state machine

A physical tile version transitions only through legal states:

```text
ABSENT
  -> READING
  -> ENCODED_HOST
  -> VERIFYING
  -> DECODED_HOST
  -> COPYING_TO_DEVICE
  -> DEVICE_RESIDENT
  -> COMPUTING / REDUCING
  -> DIRTY_DEVICE or CLEAN_DEVICE
  -> COPYING_TO_HOST
  -> DIRTY_HOST
  -> WRITING
  -> COMMITTED
  -> EVICTABLE
  -> ABSENT
```

Not every backend uses every state, but transitions MUST be explicit. State records contain version, location, lease IDs, producer event, checksum state, dirty/clean status, and reference count. A tile cannot become visible to readers before its producing event and transaction commit complete.

### 6.3 Aliases and views

Tied weights and storage aliases MUST preserve identity. The manifest describes unique storage objects separately from tensor views. Immutable aliases may share physical chunks. Mutable aliases require a single version root and conflict rules. Copy-on-write creates a new version and atomically updates the logical mapping after data commit.

Graph canonicalization MUST reject an alias/mutation pattern that cannot be represented safely by the current version model. It MUST NOT duplicate tied parameters silently.

### 6.4 Tile handles and leases

A materialized tile is accessed only through a lease that includes:

- lease ID and generation;
- resource dimensions and bytes charged;
- arena/buffer identity and bounds;
- logical tensor/version/tile coordinates;
- access mode;
- terminal event or host completion condition;
- cancellation and reclamation state.

Destruction of a language object is not lease release. A lease becomes reusable only after the terminal event has completed and all dependent tasks are terminal. Stale completion callbacks are ignored using task epoch and lease generation checks.

## 7. Storage and checkpoint subsystem

### 7.1 Store contract

Every store is range-oriented and asynchronous at the abstract interface. It advertises alignment, maximum request size, scatter/gather limits, cancellation strength, durability, direct-I/O support, mapping support, checksums, transactions, and concurrency limits.

`read_async(ranges, destination_lease)` MUST verify that requested lengths fit the destination mapping. Short reads are errors. `write_async(..., transaction_id)` writes unpublished data. `publish(transaction_id, root_record)` is the sole visibility point for mutable roots. `abort` is idempotent.

Required initial stores:

- standard file `pread`/`pwrite` adapter;
- memory-mapped immutable adapter;
- AMS packed container adapter;
- safetensors metadata/range adapter;
- bounded host cache above stores.

Direct I/O, io_uring, GPUDirect Storage, cloud/object stores, and remote tiers are optional capabilities, never assumptions of the invariant.

### 7.2 AMS package

The package consists of a canonical manifest, canonical graph IR, aligned chunk objects, optional backend-specific packs, signatures, and provenance. Parsers MUST enforce:

- maximum manifest, string, array, rank, tensor, layout, and chunk counts;
- unique IDs and valid references;
- checked non-overlapping metadata/data ranges;
- chunk bounds within declared objects;
- exact logical coverage for complete layouts, excluding declared sparse/default regions;
- codec maximum expansion and metadata validity;
- alias acyclicity or explicitly legal view graphs;
- version compatibility and required feature flags;
- content hash and signature policy.

The loader MUST parse metadata without importing Python objects or invoking checkpoint-defined code.

### 7.3 Conversion

The converter reads source metadata first, constructs a deterministic conversion plan, and processes bounded source and target ranges. It MUST NOT load the entire source model. Each output chunk is written to a temporary package, checksummed, and recorded. The canonical manifest is written last and atomically published only after complete verification.

Conversion is resumable through a journal whose entries are idempotent and content-addressed. Source provenance records source hashes, tool version, command/config digest, layouts/codecs, and license metadata. The converter runs with resource limits and, for untrusted formats, in a sandboxed process.

### 7.4 Cache

The cache stores immutable or versioned decoded/encoded tiles. Cache identity includes package root, storage object/version, layout, codec version, and logical coordinates. Eviction is lease-aware. Dirty mutable entries are not evicted before successful transactional write. Admission may bypass cache for one-shot streams. Cache accounting includes metadata, allocator granularity, and filesystem page-cache assumptions where measurable.

## 8. Canonical AMS IR and compiler

### 8.1 Capture policy

The primary PyTorch path uses a functional graph capture/export mechanism rather than recursive `nn.Module` replacement. Capture MUST preserve or explicitly model:

- functional linear projections and fused operations;
- tied parameters and views;
- buffers and mutable state;
- symbolic shapes and guards;
- control flow supported by the front end;
- device/dtype conversions;
- RNG and side effects;
- attention mask/cache semantics;
- custom operators and decompositions.

Eager module hooks may exist as compatibility helpers but are not the guaranteed invariant path.

### 8.2 IR model

Canonical IR is framework-independent and serializable. It contains:

- typed values with shape/dtype/layout domains;
- pure nodes and explicit effect tokens;
- immutable constants and virtual tensor references;
- regions/blocks and finite control-flow guards;
- alias and mutation effects;
- RNG stream operations;
- state roots and version transitions;
- operator schema/version and attributes;
- source locations and diagnostics metadata.

IR deserialization is validated and bounded. Unknown required opcodes are rejected. Optional metadata may be ignored only when its feature bit is non-semantic.

### 8.3 Pass pipeline

The compiler executes deterministic, versioned passes:

1. capture/import and metadata-only checkpoint binding;
2. schema validation and functionalization;
3. shape propagation with finite guards;
4. alias/view and effect analysis;
5. canonical decomposition into registered semantics;
6. pattern recognition for attention, MLP, MoE, convolution, and SSM structures;
7. liveness and virtual-region analysis;
8. legal fusion discovery;
9. per-node candidate generation;
10. graph-level planning and resource proof;
11. task-template lowering;
12. plan serialization and independent validation.

Every pass produces diagnostics with source graph locations. Pass outputs are cacheable by input hash, pass version, and configuration. Non-deterministic pass behavior is forbidden.

### 8.4 Operator coverage proof

For every reachable node under accepted guards, the compiler records the chosen semantic operator and implementation. Coverage is mechanically checked. A custom operator must supply shape inference, minimum working set, legal plan enumeration, a reference implementation, and numerical/resource contracts. Opaque Python callbacks, dynamically loaded kernels without workspace declarations, or data-dependent unbounded allocation are unsupported on the guaranteed path.

## 9. Planner and cost model

### 9.1 Inputs

The planner consumes:

- canonical graph and finite shape guards;
- tensor layouts/codecs and storage routes;
- backend capabilities and fixed runtime reserves;
- benchmark-calibrated bandwidth, latency, kernel, decode, and contention profiles;
- exact resource quotas;
- numerical, determinism, security, latency/throughput, energy, and wear policies;
- request envelope and phase (prefill, decode, training forward/backward/update).

Free-memory telemetry may guide cooperative mode but is not a reservation. The planner operates against capacities granted by the broker/backend reservation.

### 9.2 Candidate generation

Each operator plugin enumerates a finite candidate set. For dense linear `Y = X W^T + b`, candidates tile token/batch `P`, output `I`, and reduction `J`. A candidate records:

- `P_t`, `I_t`, `J_t`, loop order, and split-reduction topology;
- input, weight, accumulator, output, bias, scale/zero-point, packing, transfer, and workspace bytes;
- buffer depth and path;
- storage layout and codec;
- backend kernel/implementation ID;
- epilogue/fusion decisions;
- reduction order and numeric mode;
- exact tail-tile formulas;
- task template and resource vector;
- physical I/O, copies, FLOPs, launch count, and estimated time.

Candidates are rejected unless every component fits reservation and capability limits. Candidate search MUST include progressively smaller legal tiles down to a primitive fallback. Pareto dominance removes candidates that are no better in any policy-relevant cost/resource dimension.

### 9.3 Exact byte accounting

Accounting includes logical and storage dtypes, row/plane padding, alignment, allocator granularity, double/triple buffers, codec metadata, decoded expansion, scales/zero points, kernel packing, library workspace, accumulator dtype, output spill staging, event/task descriptors, and fixed context reserve. It cannot use `elements * assumed_bytes` when formats or layouts differ.

The plan validator independently recomputes formulas from serialized dimensions. Runtime debug builds compare observed high-water marks and every lease against plan bounds. A plugin that requests undeclared workspace causes `BROKER_VIOLATION` and is quarantined from guaranteed support.

### 9.4 Graph-level selection

Graph selection accounts for liveness, fusion, recomputation, spill, cache reuse, layout conversion, and phase-specific behavior. The initial implementation MAY use dynamic programming or bounded search over operator frontiers. It MUST always retain a known-safe fallback rather than time out without a capacity answer.

The objective is lexicographic unless policy says otherwise:

1. semantic/numerical validity;
2. resource feasibility and bounded progress;
3. reliability/security constraints;
4. latency or throughput;
5. energy, endurance, and secondary costs.

A fast but infeasible plan is not a candidate.

### 9.5 Plan validation and cache

A serialized plan is immutable and content-addressed. Its cache key includes model/graph/layout hashes, hardware and software fingerprints, calibration ID, request envelope, budgets, policy, and plugin/kernel versions. Runtime validation rechecks schema, signatures if required, resource proof, capabilities, fixed reserves, guards, and storage routes. Calibration drift or changed capacity causes a safe replan, not silent reuse.

## 10. Memory broker, arenas, scheduler, and executor

### 10.1 Preallocated arenas

Guaranteed fast paths use arenas reserved during preflight. Required arena classes include device, host-pinned, host-pageable staging, metadata/task, and optionally network or codec workspace. Suballocation is deterministic and bounded. Fragmentation overhead is included in the reservation.

Vendor libraries and kernel modules are warmed or queried during preflight. Their fixed context and handle allocations are measured/reserved. Kernels receive caller-provided outputs and workspace. A kernel that internally performs an unbounded allocation is not allowed in a guaranteed implementation ID.

### 10.2 Atomic admission

The scheduler maintains available credits by resource dimension. A ready task is admitted only when `try_lease(task.resource_vector)` atomically succeeds. On failure it acquires nothing and remains ready. This no-partial-hold rule prevents deadlock caused by tasks holding memory while waiting for queue, I/O, descriptor, or transaction capacity.

Fairness combines request priority, phase criticality, age, locality/reuse, and bounded starvation prevention. Cancellation and error propagation outrank speculative prefetch. Prefetch is always revocable before demand work.

### 10.3 Task model

Task kinds include READ, VERIFY, DECODE, COPY, COMPUTE, REDUCE, COMMIT, EVICT, and CONTROL. Each task contains:

- task/epoch/session IDs;
- operator/node/tile coordinates;
- predecessors and dependent count;
- idempotence class and retry policy;
- complete resource vector;
- input/output virtual versions;
- buffer binding descriptors;
- backend/store operation and attributes;
- numerical and integrity contract;
- terminal event and deadline/priority class;
- structured source location.

The task lifecycle is `CREATED -> READY -> LEASED -> SUBMITTED -> COMPLETING -> SUCCEEDED|FAILED|CANCELLED`, with a separate retry transition only for idempotent tasks and only after resources from the prior attempt are safely reclaimed.

### 10.4 Event lifetime

Host submission does not imply completion. Leases and source/destination buffers remain live until the relevant device, I/O, codec, or network terminal event signals. Completion processing validates task epoch and lease generation before state mutation. Event objects themselves are brokered and bounded.

### 10.5 Asynchronous pipeline

The optimized data path overlaps read/decode for tile `i+1`, host-to-device copy for tile `i`, and compute/reduction for tile `i-1` when capabilities, credits, and contention permit. Buffer depth is planned and fixed. The runtime degrades to double buffering or synchronous one-buffer execution without changing correctness.

Pinned memory is used only under explicit quota. `non_blocking` flags are issued only when source/destination and backend semantics support actual asynchronous transfer. Storage-to-device direct paths are optional plan alternatives and require their own alignment, registration, and lifetime contracts.

### 10.6 Progress and backpressure

The scheduler sleeps on completion, credit release, new work, cancellation, or shutdown events when it cannot submit; it does not busy-spin. Output consumers have bounded queues. When output backpressure reaches its limit, the session stops admitting downstream generation work while allowing already submitted work to complete and release resources.

A watchdog detects tasks exceeding policy timeouts but does not reclaim live device buffers until the backend confirms completion/reset. Backend loss transitions affected sessions to typed failure or a safe checkpointed fallback at a replan point.

## 11. Backend abstraction

### 11.1 Required capabilities

A backend reports devices, address spaces, allocation granularity, alignment, copy paths, stream/queue/event semantics, supported dtypes/layouts, kernels, workspace formulas, determinism, maximum dimensions, graph capture, peer access, and reset/error behavior.

Required first backends:

- CPU reference backend: scalar and bounded vector primitives, synchronous baseline, then asynchronous worker pools.
- One accelerator backend: CUDA is a practical first choice; ROCm and Metal follow through the same contract.

### 11.2 Backend reservation

`reserve_runtime(config)` creates fixed contexts, handles, streams/queues, events, and arenas. It returns a capability-stamped reservation with exact capacities. No request work begins before this succeeds. Runtime allocations outside reservation are prohibited except small language-runtime metadata explicitly bounded by configuration and covered by process memory tests.

### 11.3 Kernel contract

Each implementation ID declares:

- accepted shapes, dtypes, layouts, alignment, and tail rules;
- input/output alias restrictions;
- exact workspace formula and whether workspace may be zero;
- numerical mode and reduction order;
- deterministic support;
- output initialization/accumulation semantics;
- error behavior and asynchronous completion semantics;
- capability/version fingerprint.

Scalar reference kernels exist for every semantic operator. Accelerator differential tests cover random, boundary, non-contiguous/view-normalized, tail, NaN/Inf, and adversarial values.

## 12. Operator plugin requirements

Every plugin implements semantic inference, minimum working set, plan enumeration, task-template expansion, reference execution, numerical properties, and optional backward plugin. The following operators form the first production dense-decoder support set.

### 12.1 Linear, batched matmul, and contractions

The canonical tiled recurrence is:

```text
for output tile (P_a, I_b):
    A = initialize_bias_or_zero(P_a, I_b)
    for reduction tile J_c in declared order:
        X_tile = materialize(X[P_a, J_c])
        W_tile = materialize(W[I_b, J_c])
        A = accumulate(A, X_tile @ transpose(W_tile))
    apply legal epilogue
    commit or forward output tile
```

All three logical axes are tileable. Completed output tiles may be spilled. Output-stationary is the universal baseline; input- and weight-stationary candidates are performance alternatives. Split reduction requires planned partial accumulators and deterministic merge policy when requested.

### 12.2 Embedding and tied LM head

Embedding reads only referenced row tiles, deduplicates indices under a bounded table, preserves output order, and handles invalid IDs according to model semantics. Tied output weights share the same storage object and version.

The LM head streams vocabulary blocks. Greedy decoding maintains `(max_value, lowest_index_on_tie)` according to framework tie rules. Top-k maintains a bounded deterministic heap/selection structure. Softmax sampling uses online normalization and a declared RNG stream; processors requiring the full logits vector either receive a virtual logits interface, trigger external materialization under a proven spill plan, or are rejected.

### 12.3 Normalization

LayerNorm uses a stable one-pass Welford or two-pass recurrence over feature tiles. RMSNorm accumulates scaled squares. Parameters stream by feature tile. The planner records reduction dtype, epsilon placement, and finite/non-finite policy. Output tiles may be recomputed or spilled when the input cannot remain resident between statistics and normalization passes.

### 12.4 RoPE and positional transforms

Rotary transforms tile token/head/feature dimensions while preserving paired-coordinate semantics. Sin/cos tables are generated or streamed with bounded cache. Position scaling and model-specific layouts are explicit attributes and conformance-tested.

### 12.5 Gated MLP

Up and gate projections may be fused only when intermediate tiles fit the planned frontier. The activation is applied tilewise. The down projection reduces across intermediate tiles without materializing the full hidden expansion. Exact nonlinear semantics and rounding boundaries are declared.

### 12.6 Exact attention

For each query tile, stream visible KV blocks in logical order and maintain rowwise maximum `m`, normalizer `l`, and output accumulator `O` using online softmax. The mask function receives logical coordinates and must cover causal, padding, sliding-window, prefix, and model-specific patterns. All-masked-row behavior follows the imported framework/model contract.

Score tiles, probability tiles, and decoded KV tiles are bounded workspaces. The query tile, `(m,l,O)`, and one or more pipelined KV blocks form the resident frontier. KV/output may exceed every fast tier and reside in virtual pages.

### 12.7 Virtual KV cache

KV is a logical sequence of versioned pages by request, layer, head grouping, and token range. Page size is plan-selected. Prefix sharing increments immutable page references. Appending to a shared partial page performs bounded copy-on-write. Data and checksum commit before the logical length/mapping is atomically published.

Eviction is independent of logical validity. Pages may move among device, host, and storage while references remain stable. Beam reorder changes mappings rather than copying complete KV. Page tables, refcounts, and journals have bounded metadata or are themselves paged for extreme contexts.

### 12.8 MoE

Router logits and top-k expert selection stream without full expert residency. Token-to-expert assignments are stored in bounded or spilled routing buckets. Experts execute sequentially or in planned groups. Output accumulation preserves token order and combines expert weights under the numerical contract. The worst-case routed-token bound comes from request envelope and expert capacity policy.

### 12.9 Convolution

Convolution plans tile batch, spatial, input-channel, and output-channel dimensions. Halo regions and overlap are explicit. Direct, im2col-like, FFT, or backend kernels are candidates only with bounded workspace. The scalar fallback indexes source and weight elements directly.

### 12.10 State-space and recurrent operators

Selective scans and recurrent models expose finite carry state. Plans tile batch/channel/time and either stream the recurrence sequentially or use block summaries with bounded merge state. Data-dependent parameter generation is decomposed into registered operators; opaque fused kernels need declared workspace and semantics.

### 12.11 Views, concat, slicing, and indexing

Metadata-only views do not copy when physical layouts support the requested access. Otherwise the compiler inserts a bounded relayout task. Concatenation is a virtual region map when possible. Gather/scatter operations require finite index bounds and conflict semantics. Dynamic indices are validated before computing byte ranges.

## 13. Inference engine lifecycle

### 13.1 Cold start

Cold start validates package/graph/tokenizer hashes, opens bounded storage handles, reserves backend runtime, loads or creates a plan, preallocates arenas, warms selected kernels within reserve, and emits a cold-start trace. It never loads all weights.

### 13.2 Prefill

Prefill uses token tiles and may select larger matrix/attention blocks. Activation liveness is compiled. Weight reuse and KV writes are scheduled under the same broker. Prompt length is bounded by the accepted envelope. Prefix caches are integrity/version checked before reuse.

### 13.3 Decode

Decode normally uses a small token dimension and benefits from weight-stationary reuse across continuous-batch requests. The scheduler may cohort compatible requests by plan/model/layer while preserving quotas and cancellation isolation. A request cannot pin a weight indefinitely; cache residency is revocable.

### 13.4 Sampling and output

Sampling is deterministic when requested and records RNG counters. Output token/text queues are bounded. Tokenizer decoding may run in a separate bounded worker pool. User callbacks execute outside runtime locks and cannot allocate from guaranteed arenas.

### 13.5 Durable sessions

Optional durable sessions checkpoint token position, RNG, stop automaton, KV mappings/version roots, generation policy, output cursor, and plan/model versions transactionally. Restart validates that immutable model roots match. A partially published token or KV append is resolved through journal state; no duplicate logical token is emitted.

## 14. Training and fine-tuning extension

Training is a separate advertised capability and cannot be inferred from inference support.

### 14.1 Backward lowering

Forward and backward are jointly compiled. For linear operations:

- `dX += dY W` tiles output/reduction/input dimensions;
- `dW += dY^T X` tiles parameter dimensions and accumulates gradient tiles;
- `db += reduce(dY)` uses bounded reductions.

Gradient and activation tensors are virtual. The planner chooses spill versus rematerialization using liveness, I/O, compute, and deterministic RNG constraints.

### 14.2 Activation strategy

Every saved activation has a policy: retain, spill encoded/decoded, recompute from a checkpoint, or discard. Rematerialization regions record input versions, RNG state, and effect boundaries. Stateful or externally visible effects cannot be replayed unless explicitly idempotent and versioned.

### 14.3 Optimizer streaming

Optimizer state and master weights are tiled. A step transaction reads parameter, gradient, and state tiles; computes updated versions; writes checksummed new tiles; and records journal completion. The new root is published only when every tile and step metadata are durable. Interrupted steps resume or abort without exposing a mixed version.

Global norm clipping uses a streaming reduction followed by the update pass. Gradient accumulation and microbatch counts are finite and recorded. Loss scaling, scheduler state, data position, RNG, and distributed epoch are part of the checkpoint root.

### 14.4 Distributed training

Data, tensor, pipeline, and state sharding may compose with AMS. Collective workspaces and in-flight messages are resources in the same admission vector. Collectives cannot allocate hidden scratch outside declared bounds. Failure semantics and restart consistency are explicit for each distributed mode.

## 15. Multi-device and distributed composition

The topology model represents devices, NUMA nodes, storage controllers, PCIe/root complexes, peer links, and network paths with capacities and calibrated costs. A virtual tensor layout may have replicated, sharded, erasure-protected, or cached regions.

Linear parallelization candidates include output sharding, reduction sharding plus all-reduce/reduce-scatter, token sharding, and heterogeneous tile assignment. The planner includes collective memory and communication. Slow devices may receive fewer tiles; correctness does not require homogeneous hardware.

A local-only invariant may use multiple devices in one machine. Remote execution is an optional backend/tier and MUST be disclosed because it changes privacy, availability, and locality assumptions.

## 16. Error, cancellation, retry, and recovery model

### 16.1 Stable error categories

At minimum:

- `PREFLIGHT_NO_BACKING`
- `PREFLIGHT_NO_WORKING_SET`
- `UNSUPPORTED_OP`
- `INVALID_PACKAGE`
- `INTEGRITY_FAILURE`
- `SIGNATURE_FAILURE`
- `CAPABILITY_MISMATCH`
- `PLAN_INVALID`
- `RESERVATION_LOST`
- `BROKER_VIOLATION`
- `IO_FAILURE`
- `BACKEND_FAILURE`
- `NUMERIC_FAILURE`
- `TRANSACTION_FAILURE`
- `DEADLINE_EXCEEDED`
- `CANCELLED`
- `INTERNAL_INVARIANT`

Payloads include phase, subsystem, retriable flag, model/plan/session/task correlation, resource evidence, causal chain, fallback attempts, and operator/source location. Secrets and raw user data are redacted by policy.

### 16.2 Cancellation

Cancellation is cooperative but prompt. It prevents new non-cleanup tasks, propagates through dependency graphs, cancels queued I/O where supported, and lets submitted device work reach a safe terminal event. Leases are reclaimed only after completion. Mutable transactions are aborted or completed according to idempotence and journal state. Session close is idempotent.

### 16.3 Retry

Immutable range reads, verification, decode, pure compute, and content-addressed writes may be retried under bounded policy. Non-idempotent publish, output emission, and external side effects require transaction IDs or exactly-once records. Ambiguous completion consults the journal before retry.

### 16.4 Recovery

The recovery tool scans package/session/training journals, validates checksums, completes or rolls back transactions, repairs orphan temporary objects, and reports leaks. It never guesses which root is current; publication records and monotonic versions decide.

## 17. Security model

### 17.1 Threat model

Treat model packages, manifests, graph IR, tokenizer assets, codecs, plugins, and traces as untrusted unless verified. Threats include parser memory corruption, integer overflow, path traversal, decompression bombs, malicious dimensions, overlapping chunks, signature confusion, code execution through checkpoint formats, GPU kernel faults, symlink races, quota exhaustion, data leakage through logs/cache, and unsigned plugin substitution.

### 17.2 Required controls

- Metadata-only safe loading; never unpickle untrusted checkpoints in the main process.
- Bounded parsers with checked arithmetic, recursion/depth limits, and canonicalization.
- Sandboxed conversion for unsafe source formats.
- Allowlisted codecs/operators/plugins with version and signature policy.
- Content hashes for every immutable chunk; signature verification according to deployment policy.
- Path confinement and no manifest-controlled arbitrary absolute path access.
- Per-tenant storage, pinned-memory, device, queue, and task quotas.
- Zeroization policy for sensitive mutable buffers and isolation-aware cache keys.
- Structured redaction for prompts, tokens, tensor values, credentials, and paths.
- Dependency lockfiles, SBOM, provenance, signed release artifacts, and vulnerability response process.
- Fuzzing for manifest, graph, plan, tokenizer/config, journal, and codec parsers.

Third-party native plugins run out of process or are explicitly trusted; they do not inherit guaranteed status merely by implementing an interface.

## 18. Observability and operations

### 18.1 Metrics

Expose bounded-cardinality metrics for:

- bytes resident, leased, reserved, spilled, and high-water by tier/arena/class;
- read/write/copy/compute/decode volume and duration;
- queue depth, ready time, credit wait, scheduler decisions, and starvation age;
- cache hit/miss/eviction and reuse distance;
- task outcomes, retries, cancellations, fallback/replan counts;
- per-token prefill/decode latency, time to first token, throughput;
- storage bandwidth/latency, device utilization, and overlap efficiency;
- checksum/signature failures, broker violations, and journal recovery;
- plan-estimate error and calibration age.

Labels use stable IDs or bounded enums, not arbitrary tensor names or user text by default.

### 18.2 Traces

Every task emits timestamps for ready, leased, submitted, started where observable, and terminal. Trace events record resource deltas, tile coordinates (optionally hashed/redacted), path, implementation ID, dependencies, and error. The trace schema supports replay of scheduling decisions without model data.

### 18.3 Tools

Required CLIs:

- `ams-convert`: safe streaming conversion;
- `ams-inspect`: manifest/graph/layout/operator report;
- `ams-preflight`: capacity and plan report without execution;
- `ams-run`: controlled generation/inference;
- `ams-trace`: summarize timeline, overlap, stalls, and high-water marks;
- `ams-benchmark`: reproducible benchmark harness;
- `ams-recover`: journal inspection and recovery;
- `ams-validate`: package, schema, signature, and resource-proof validation.

Every command supports machine-readable JSON output, stable exit codes, config digest, and `--dry-run` where meaningful.

## 19. Configuration, compatibility, and versioning

Configuration is schema-validated before resource reservation. Unknown fields are errors by default. Environment-variable overrides are allowlisted and included in the effective-config digest. Secret values are references to secret providers, not serialized plaintext.

The package format, IR, plan, trace, plugin ABI, and public API have independent semantic versions. Major versions break compatibility; minor versions add backward-compatible optional features; patch versions fix behavior without changing contracts. Readers reject unknown required feature flags and may ignore unknown optional metadata.

Plans are not portable across incompatible capabilities or changed fixed reserves. Package data may be portable while backend-specific packs are optional. Migration tools operate transactionally and preserve provenance.

## 20. Testing strategy

### 20.1 Unit and property tests

Required properties include:

- tile partitions cover exactly the logical domain with no overlap/gap unless declared;
- byte ranges never overflow and remain within objects/buffers;
- resource-vector atomic acquisition never goes negative;
- lease/event state machines reject illegal transitions and stale completions;
- manifest aliases and layouts validate;
- canonical serialization is stable and round-trips;
- online softmax combines blocks associatively over real/reference arithmetic;
- streamed top-k/greedy matches full-vector tie behavior;
- transaction publish exposes either old or new root, never a mix;
- cancellation eventually reclaims every lease after terminal events.

Use property-based generators for shapes, tail tiles, layouts, codecs, task DAGs, cancellation points, and failures. Model checking or systematic schedule exploration SHOULD cover the broker/scheduler state machine at small bounds.

### 20.2 Differential tests

Every semantic operator compares:

1. exact/scalar reference;
2. CPU tiled implementation across tile sizes and loop orders;
3. accelerator implementations across supported dtypes/layouts;
4. fused versus unfused paths;
5. spill/reload versus resident execution;
6. deterministic versus permitted fast mode.

Test random values and adversarial cases: zero extents where legal, one-element dimensions, prime dimensions, all tail tiles, extreme values, cancellation after every task boundary, corrupt/short I/O, NaN/Inf, all-masked attention rows, duplicate embedding IDs, tied weights, and shared KV pages.

### 20.3 Memory invariant tests

Tests MUST use hard arena quotas or isolated devices/process limits, not only telemetry. Required oversize cases:

- checkpoint larger than device arena;
- checkpoint larger than host cache/DRAM budget and served from file/NVMe;
- one weight row larger than device arena, proving reduction-axis tiling;
- output tensor larger than device and host arenas, proving output tiling/spill;
- activation frontier larger than device arena;
- KV state larger than device and host cache;
- codec expansion and tail workspace at limits;
- concurrent sessions whose sum fits only under atomic quotas;
- external pressure/reservation loss producing a structured replan/failure, not allocator crash.

Instrument all managed allocators and kernel workspace. CI fails on an unaccounted allocation above a small explicitly configured metadata tolerance.

### 20.4 Fault injection

Inject short reads/writes, checksum mismatch, ENOSPC, permission loss, descriptor exhaustion, cancellation, delayed completion, stale event, device reset, kernel error, codec failure, journal corruption, process kill before/after data write and before/after root publish, plugin over-allocation, and calibration mismatch. Verify typed errors, no lease leaks, root consistency, and safe restart.

### 20.5 Model integration matrix

Start with small synthetic transformers whose dimensions force every tiling path, then one declared decoder family in several sizes/dtypes. Add architecture families only after operator coverage artifacts pass. For every advertised model, publish:

- exact model/config/tokenizer hash;
- graph signature and operator coverage;
- plan/config/hardware/calibration fingerprints;
- output differential results;
- memory high-water proof;
- performance trace and baseline comparison.

### 20.6 Performance protocol

Report time to first token, inter-token latency distribution, prompt/decode throughput, physical bytes read/written, transfer volume, cache hit rate, overlap, device/CPU utilization, energy where available, and storage write amplification. Compare against resident execution when it fits and relevant offload/runtime baselines. Separate cold and warm cache. Never extrapolate unsupported model sizes as measured results.

## 21. CI/CD and release engineering

Every pull request runs formatting, lint, strict typing, schema validation, architecture checks, licenses, docs, CPU unit/property/differential/invariant tests, deterministic serialization, sanitizer builds where applicable, and a small end-to-end external-memory model.

Scheduled self-hosted CI runs accelerator backends, direct-I/O paths, multi-device, long stress, fault recovery, parser fuzz corpora, and performance regression. Hardware pools are fingerprinted; noisy benchmarks use repeated runs and robust statistics.

Release CI:

1. builds from a signed immutable tag in clean environments;
2. runs the complete conformance matrix;
3. generates packages/wheels/native artifacts with reproducible metadata;
4. emits SBOM and SLSA-style provenance where supported;
5. signs artifacts and publishes checksums;
6. runs upgrade/downgrade and old-package compatibility tests;
7. smoke-installs and executes CPU fallback and at least one accelerator path;
8. publishes known limitations, supported operators/models/backends, and benchmark artifacts.

A performance baseline update requires rationale, traces, before/after memory proofs, and reviewer approval. A faster path that exceeds resource bounds is rejected.

## 22. Implementation order and acceptance gates

### Phase 0: executable specification skeleton

Implement schemas, typed errors, immutable descriptors, canonical serialization, CPU scalar primitives, a synchronous file store, simple broker, and exact linear/reduction/elementwise reference execution. Gate: a tensor larger than the configured arena executes with exact parity and no unaccounted allocation.

### Phase 1: CPU external-memory engine

Implement AMS container/safetensors range access, streaming converter, versioned virtual tensors, preallocated host arenas, leases/events, task DAG, asynchronous worker pools, transactional writes, and trace tooling. Gate: a small transformer whose weights, activation, and output independently exceed the host arena executes from storage.

### Phase 2: first accelerator backend and true 2D AMS linear

Implement fixed device arena, pinned ring, streams/events, async copy, kernel registry, exact resource formulas, double/triple buffering, and 2D output/reduction tiling. Gate: both a single weight row and output exceed device arena; resident, spill, and scalar paths match reference.

### Phase 3: dense decoder inference

Implement embedding/tied head, normalization, RoPE, gated MLP, exact attention, virtual KV, vocabulary streaming, deterministic sampling, compiler import, preflight, and session API. Gate: supported decoder models run when checkpoint exceeds VRAM and in at least one test exceeds the host cache budget.

### Phase 4: planner and throughput

Implement calibration, candidate frontiers, graph liveness/fusion, separate prefill/decode plans, cache policy, continuous batching, trace analysis, and optional direct I/O. Gate: resource estimates match observed high water exactly within declared allocator metadata; performance regressions are gated.

### Phase 5: model breadth and plugin SDK

Add MoE, convolution, SSM/recurrent, dynamic shapes, second accelerator backend, distributed topology, and third-party plugin conformance. Gate: every advertised model produces a complete operator coverage and memory proof artifact.

### Phase 6: training

Implement joint backward lowering, activation spill/rematerialization, virtual gradients, streamed optimizer, transactional checkpoints, and restart. Gate: gradient/optimizer differential tests and kill-point restart tests pass.

### Phase 7: enterprise hardening

Complete multi-tenant quotas, signatures, sandboxing, fuzzing, recovery, signed packages, SBOM/provenance, upgrade tests, stress, incident runbooks, and operational SLOs. Gate: all rows in `ACCEPTANCE_MATRIX.md` required for the target release are evidenced.

## 23. Definition of done

A subsystem is complete only when it has:

- a versioned contract and explicit invariants;
- bounded implementation with checked arithmetic and ownership;
- unit, property, differential, cancellation, and fault tests as applicable;
- high-water/resource instrumentation and structured errors;
- public or operator documentation and at least one executable example;
- compatibility and migration policy;
- calibration/benchmark coverage when on a critical path;
- security review for parsing, native code, storage, codecs, plugins, or tenant isolation;
- no placeholder, silent exception, unbounded convenience allocation, or bypass of the broker in a release path.

The repository may label experimental features, but experimental paths are feature-gated, excluded from the guaranteed support matrix, and cannot be selected silently by production policy.

## 24. Prohibited shortcuts

The following implementations fail the AMS contract:

- replacing only `nn.Linear` modules while missing functional operators, tied weights, views, or custom graph paths;
- tiling output rows only, leaving one row/reduction slice or full output as an unbounded allocation;
- pinning the complete input or output when those tensors may exceed the tier;
- calculating tile size from a fraction of transient free VRAM without reserving all live buffers/workspaces;
- calling `.cuda()` per tile and relying on deletion/caching allocator behavior;
- issuing nonblocking copies from pageable memory and assuming overlap;
- permitting vendor kernels or plugins to allocate undeclared workspace;
- keeping KV cache, logits, routing buffers, gradients, or optimizer state outside the virtual tensor model;
- retrying non-idempotent commits or output emission without a transaction record;
- using quantization or approximation as an undisclosed prerequisite for capacity;
- claiming support without a mechanically complete operator coverage report;
- publishing benchmark numbers without configurations, traces, fingerprints, and memory evidence.

These constraints are release blockers, not recommendations.
