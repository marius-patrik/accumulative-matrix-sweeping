# AMS Acceptance Matrix

This matrix converts the AMS invariant into executable release gates. A row is complete only when the linked automated test, trace, resource report, and failure evidence are retained as CI artifacts. `P0` through `P7` refer to the implementation phases in `IMPLEMENTATION_SPEC.md`.

## Evidence requirements

Every required row produces:

- test binary/package and source revision;
- effective configuration and policy digest;
- model/package/graph/plan hashes;
- hardware, software, and calibration fingerprints;
- trace bundle conforming to `schemas/trace.schema.json`;
- allocator/arena high-water report;
- expected and observed semantic result;
- typed error payload for negative tests;
- pass/fail decision with no manual reinterpretation.

## Core invariant and preflight

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| INV-001 | P0 | Finite external-memory execution | Scalar linear/reduction where source and output exceed the configured arena | Exact result; peak managed memory remains within reservation |
| INV-002 | P1 | Model exceeds host arena | Small transformer checkpoint larger than host cache, read from file store | Complete inference; no full checkpoint materialization |
| INV-003 | P2 | Model exceeds device arena | Accelerator run with checkpoint larger than VRAM reservation | Complete inference; no allocator OOM on managed path |
| INV-004 | P2 | One weight row exceeds device arena | Choose `in_features * storage_bytes` greater than device arena | Reduction-axis tiling selected; exact/reference parity |
| INV-005 | P2 | Output exceeds device arena | Output tensor larger than device arena | Output-axis tiling/spill; exact/reference parity |
| INV-006 | P1 | Output exceeds host arena | CPU path with output larger than host arena | Bounded output tiles committed to backing store |
| INV-007 | P3 | KV exceeds device arena | Long-context decode with KV larger than device arena | KV pages tiered; token parity with reference |
| INV-008 | P3 | KV exceeds host cache | KV larger than device plus host cache budget | Storage-backed KV continues; bounded metadata and buffers |
| INV-009 | P0 | No backing capacity | Quota smaller than package plus worst-case spill | Preflight rejects `PREFLIGHT_NO_BACKING` before execution |
| INV-010 | P0 | No legal working set | Set every backend quota below runtime base plus primitive | Preflight rejects `PREFLIGHT_NO_WORKING_SET` before execution |
| INV-011 | P3 | Unsupported reachable op | Insert registered graph node without plugin/fallback | Preflight rejects `UNSUPPORTED_OP` with source location |
| INV-012 | P3 | Unreachable unsupported branch | Guard proves branch unreachable for accepted envelope | Plan may accept; proof records guard and excluded branch |
| INV-013 | P2 | CPU fallback | Disable/undersize accelerator while CPU primitive fits | Plan selects CPU fallback without numerical downgrade |
| INV-014 | P2 | Policy disables fallback | CPU fallback fits but policy forbids it | Typed policy/preflight rejection; no false universal claim |
| INV-015 | P4 | Plan replay validation | Change capability/fixed reserve after plan creation | Cached plan rejected and safely replanned |

## Resource accounting and broker

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| MEM-001 | P0 | Checked byte arithmetic | Generate overflowing shape/stride/offset products | Validation rejects before allocation or I/O |
| MEM-002 | P2 | Exact resource formula | Compare plan bytes with all observed leases and workspaces | No observed lease exceeds formula; high water within reserve |
| MEM-003 | P2 | Tail-tile accounting | Prime dimensions and all partial tails | Correct bytes and results for every tail combination |
| MEM-004 | P2 | Codec expansion accounting | Encoded tile with maximum declared decode expansion | Fits declared arena; no hidden staging allocation |
| MEM-005 | P2 | Packing/workspace accounting | Kernel requiring alignment/padding/pack buffer | All overhead appears in resource proof |
| MEM-006 | P1 | Atomic vector admission | Two tasks each need complementary scarce resources | No partial hold/deadlock; one task progresses |
| MEM-007 | P1 | No negative credits | Random concurrent acquire/release schedules | Credits never negative or above capacity |
| MEM-008 | P1 | Event-bound lease lifetime | Delay device/I/O completion after host submission | Buffer is not reused until terminal event |
| MEM-009 | P1 | Stale completion protection | Reuse task slot with incremented epoch, deliver old completion | Old completion cannot mutate new task/lease |
| MEM-010 | P2 | Plugin over-allocation | Test implementation requests undeclared workspace | `BROKER_VIOLATION`; path removed/quarantined |
| MEM-011 | P2 | Fixed backend reserve | Warm libraries/handles before request and measure | Context allocations included in reservation |
| MEM-012 | P4 | Concurrent session quotas | Admit requests whose sum is near capacity | Atomic fair admission; no oversubscription |
| MEM-013 | P4 | Cooperative pressure | Revoke cooperative headroom at safe point | Controlled shrink/replan or `RESERVATION_LOST`, no crash |
| MEM-014 | P4 | Metadata tolerance | Instrument process allocations under stress | Unbrokered material allocation below declared tiny tolerance |

## Storage, package, and transactions

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| STO-001 | P1 | Bounded range read | Read a tile from a very large object | Only requested ranges and bounded readahead are issued |
| STO-002 | P1 | Short read | Inject EOF/short completion | Typed `IO_FAILURE`; no partial tile publication |
| STO-003 | P1 | Checksum failure | Corrupt one immutable chunk | `INTEGRITY_FAILURE`; no compute consumes corrupt tile |
| STO-004 | P1 | Manifest bounds | Oversized strings/counts/rank/nesting | Parser rejects within bounded memory/time |
| STO-005 | P1 | Range overflow/overlap | Malicious offsets, lengths, reserved-range overlaps | Package rejected before I/O |
| STO-006 | P1 | Alias identity | Tied embedding/LM-head storage | One storage object; both logical tensors resolve identically |
| STO-007 | P1 | Streaming conversion | Source checkpoint larger than converter budget | Successful deterministic package; bounded high water |
| STO-008 | P1 | Converter kill/restart | Kill after each journaled chunk | Resume or clean abort; no published incomplete root |
| STO-009 | P1 | Root atomicity | Kill before and after root publication | Readers see complete old or complete new root only |
| STO-010 | P6 | Optimizer transaction | Kill at every parameter-tile update boundary | Restart exposes prior or complete new step, never mixed state |
| STO-011 | P3 | KV copy-on-write | Fork prefix, append to shared partial page | Original prefix unchanged; new mapping published after commit |
| STO-012 | P3 | KV version recovery | Kill during page append | Logical length does not include uncommitted data |
| STO-013 | P7 | Signature policy | Valid, invalid, missing, and wrong-key signatures | Policy-consistent accept/reject with typed evidence |
| STO-014 | P7 | Path confinement | Manifest attempts traversal/symlink escape | Rejected; no access outside configured roots |
| STO-015 | P4 | Direct-I/O fallback | Misaligned path or unsupported filesystem | Planner selects buffered path or rejects explicitly |

## Compiler and operator coverage

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| CMP-001 | P3 | Functional capture | Model uses functional linear instead of module class | Operator captured and lowered |
| CMP-002 | P3 | Tied weights/views | Shared storage with views/offsets | Alias semantics preserved; no silent copy |
| CMP-003 | P3 | Mutation analysis | Unsafe overlapping mutation pattern | Compiler rejects with precise source diagnostic |
| CMP-004 | P3 | Symbolic finite guards | Dynamic shape with accepted upper bounds | Plan records guards; out-of-guard request rejected/replanned |
| CMP-005 | P3 | Unbounded dynamic state | Loop/request with no finite envelope | Preflight rejects before execution |
| CMP-006 | P3 | Coverage completeness | Enumerate every reachable IR node | Exactly one selected semantic implementation per node |
| CMP-007 | P3 | Deterministic compilation | Compile identical inputs in isolated runs | Byte-identical canonical IR and plan |
| CMP-008 | P4 | Plan cache key | Vary one fingerprint/policy/layout field at a time | Cache miss/revalidation occurs where required |
| CMP-009 | P4 | Fusion legality | Compare fused and unfused under alias/numeric constraints | Fusion selected only when semantics/resource bounds hold |
| CMP-010 | P5 | Custom plugin conformance | Third-party plugin candidate suite | Shape/resource/reference/fault tests pass before support |

## Numerical and semantic conformance

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| NUM-001 | P0 | Linear real/reference correctness | Random matrices across tile/loop orders | Exact scalar/reference equality |
| NUM-002 | P2 | Floating reduction modes | Compare reference, deterministic, stable, fast | Results satisfy declared bitwise/tolerance contract |
| NUM-003 | P3 | Online softmax | Random block partitions and masks | Matches full reference within mode tolerance |
| NUM-004 | P3 | All-masked attention row | Model/framework-specific all-masked case | Exact declared behavior; no accidental divide-by-zero policy |
| NUM-005 | P3 | Normalization stability | Extreme magnitudes and long feature dimensions | Stable recurrence and tolerance contract pass |
| NUM-006 | P3 | Greedy tie break | Equal maximum logits across different blocks | Matches framework lowest/declared index rule |
| NUM-007 | P3 | Streaming top-k | Adversarial ties and block boundaries | Same ordered top-k as full-vector reference |
| NUM-008 | P3 | Sampling RNG | Same seeds/counters across tile sizes | Deterministic mode emits identical token sequence |
| NUM-009 | P3 | Non-finite policy | NaN/Inf in scores, norms, logits, and weights | Policy-consistent propagation or `NUMERIC_FAILURE` |
| NUM-010 | P3 | Quantization opt-in | Capacity pressure with approximate mode disabled | No silent quantization; exact path or rejection |
| NUM-011 | P6 | Gradient correctness | Finite differences and resident autograd oracle | Gradients within declared contract across spill/recompute |
| NUM-012 | P6 | Optimizer correctness | Tiled versus resident small-model steps | Parameter/state parity within declared contract |

## Runtime, cancellation, fairness, and recovery

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| RUN-001 | P1 | Scheduler progress | Finite legal DAG with bounded delayed completions | All tasks reach terminal state; no deadlock |
| RUN-002 | P1 | Idle behavior | No ready/admissible work | Scheduler waits on event; no busy-spin |
| RUN-003 | P1 | Cancellation at every boundary | Cancel before/after each task transition | No new work after policy point; every lease reclaimed |
| RUN-004 | P2 | Cancel submitted device work | Long kernel/copy then cancel | Reclaim only after terminal event; no use-after-free |
| RUN-005 | P1 | Idempotent retry | Inject transient immutable-read/decode failure | Bounded retry succeeds or returns typed terminal error |
| RUN-006 | P1 | Ambiguous commit | Lose completion acknowledgment after write/publish | Journal consulted; no duplicate/mixed publication |
| RUN-007 | P4 | Fairness | Sustained high-priority and low-priority requests | Priority honored with configured starvation bound |
| RUN-008 | P4 | Cache eviction under lease | Evict tile selected while task still depends on it | Eviction deferred; no invalid access |
| RUN-009 | P4 | Output backpressure | Consumer stops reading bounded output queue | Generation admission pauses; resources remain bounded |
| RUN-010 | P4 | Replan safe point | Capacity/capability changes mid-session | Replan only at declared safe point with state consistency |
| RUN-011 | P3 | Durable session restart | Kill around token/KV/output commits | No duplicate/lost committed logical token |
| RUN-012 | P7 | Long-duration stress | Multi-session churn, cancellation, cache pressure | No leak, deadlock, corruption, or bound drift |

## Pipeline and performance integrity

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| PERF-001 | P2 | Real asynchronous copy | Pinned source and capable device path | Trace shows overlap when selected; otherwise plan reports no overlap |
| PERF-002 | P2 | Pageable fallback | Pageable source with nonblocking flag | Runtime does not claim unsupported overlap |
| PERF-003 | P2 | Buffer-depth bound | Double/triple-buffer plans under delayed stages | In-flight buffers never exceed planned depth |
| PERF-004 | P4 | Cost estimate calibration | Operator microbenchmarks on fingerprinted hardware | Estimate error within published threshold/confidence |
| PERF-005 | P4 | End-to-end stage attribution | Cold/warm prefill and decode | Trace accounts for wall time without unexplained serialization |
| PERF-006 | P4 | Dense read floor visibility | Cold exact dense decode with cache disabled | Physical weight reads reported; no impossible throughput claim |
| PERF-007 | P4 | Resident fast path | Model/layer fits selected tier | Planner may choose whole-layer tile without AMS overhead dominance |
| PERF-008 | P4 | Baseline reproducibility | Run resident/offload baselines with pinned configs | Artifacts include versions, configs, repetitions, and raw samples |
| PERF-009 | P7 | Regression gate | Introduce synthetic performance regression | CI blocks beyond noise-qualified threshold |
| PERF-010 | P7 | Memory-over-speed rule | Candidate is faster but exceeds resource proof | Candidate rejected; release cannot waive invariant silently |

## Security and supply chain

| ID | Phase | Requirement | Test construction | Required result |
|---|---:|---|---|---|
| SEC-001 | P1 | No unsafe checkpoint code execution | Malicious pickle/source checkpoint | Main runtime never executes it; converter sandbox rejects/contains |
| SEC-002 | P7 | Parser fuzzing | Continuous manifest/IR/plan/journal/codec fuzz corpus | No memory safety issue, hang, or unbounded allocation |
| SEC-003 | P7 | Decompression bomb | Codec metadata/content with extreme expansion | Rejected by declared maximum before materialization |
| SEC-004 | P7 | Tenant quota isolation | Competing tenants exhaust cache/pinned/device credits | Quotas enforced; no cross-tenant starvation beyond policy |
| SEC-005 | P7 | Data redaction | Sensitive prompt/path/tensor content during failures | Logs/traces redact according to policy |
| SEC-006 | P7 | Cache isolation | Same logical IDs under different tenant/package roots | No unintended data reuse or disclosure |
| SEC-007 | P7 | Plugin trust | Unsigned/unallowlisted native plugin | Not loaded on guaranteed path |
| SEC-008 | P7 | Release provenance | Build signed tag in clean environment | SBOM, checksums, provenance, and signatures published |

## Release declaration

A release support statement MUST enumerate:

- completed acceptance rows and links to evidence;
- supported package/IR/plan schema versions;
- supported operators, model graph signatures, numerical modes, backends, operating systems, and storage adapters;
- required minimum working sets by backend class;
- known unsupported operators and request features;
- measured performance configurations, separated from capacity guarantees;
- security and recovery limitations.

A row may be waived only for an experimental feature that is disabled by default and excluded from the advertised guaranteed support matrix. Core invariant rows `INV-001` through `INV-015` cannot be waived for a production release claiming bounded-residency execution.
