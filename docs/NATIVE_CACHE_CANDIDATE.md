# Experimental native checkpoint retention

This local candidate is opt-in: `--prompt-cache-native MiB` on the single-GPU
CUDA/HIP text path. `0` (the default) keeps the existing path. The shared Metal
driver rejects a positive value; SYCL uses a separate driver overlay. One
eligibility predicate gates request retention, trimming and outgoing parking,
so an outgoing image-bearing state keeps its configured stock parking floor. For example,
`--prompt-cache-native 1024` admits optional copies only when the recipient's
whole checkpoint chain, including conservative allocation overhead, fits
1024 MiB. This is an admission ceiling, not permission to discard mandatory
upstream state or explicit pins when those alone exceed it. The existing
`--prompt-cache` count cap and parked conversation byte/count caps still apply.
No managed profile enables this candidate.

The change retains compatible earlier checkpoints when a main session is
copied into a batch slot and when a slot is copied back. A copy is useful only
alongside the same branch's backing KV through the checkpoint position. Donors
keep independent state; a BYIELD transfers ownership where the main read ends.
Park/restore retains the existing ConversationCache and SavedConversation
structures. No new side pool or client key is introduced.

At admission the scheduler compares the actual input IDs with consumed IDs
kept by the main session, valid slots and parked conversations. The maximum
observed common prefix L is a scheduling hint, not a restorable state. When
R < L, the engine reads from its valid restore point R to L, saves an evictable
checkpoint there after successful exact completion, then reads the suffix.
The next fork may reuse L if the checkpoint and KV survive. No future or
interrupted state is fabricated. The last token stays in the decode window.
There is no speculative tail ladder and no automatic hard pin.

Optional snapshot publication is transactional. Allocation of the fresh snapshot,
eviction metadata and recipient directory completes before any old state is
removed. `bad_alloc` skips optional work without changing the old chain, pins or
count. Mandatory errors and GPU save/synchronization failures remain errors.

Optional allocation is declined when memory telemetry is unknown or the
available physical/cgroup room would cross max(configured floor, 8 GiB).
Other slots, parked snapshots, retained KV and pooled buffers are already
charged in that host reading; the copy estimate also charges the transient
recipient allocation. Optional checkpoints are reclaimed at request admission
under pressure, and bypass the spare-buffer pool when discarded. Sampling is
not an atomic reservation against other processes. Ordinary upstream state
and explicit pins keep their existing semantics. The runtime-only soft mark
is preserved by in-memory parking; it is not a new session-file field.

`STRATA_TRACE=1` reports N, maximum observed L, actual read_from R, the scheduled
cut, chain length/bytes and additional checkpoints retained by slot handovers.
These counts do not measure the complete user task's critical path. Existing
request timing, consumed token traces and client tool timing are still needed.

CPU policy/ownership tests use the real checkpoint/cache types, two owners,
park/borrow/take, cancellation boundaries, memory pressure and allocation
failure. An extracted actual driver boundary is tested with an allocation failure
at every observed allocation point; GPU operations are mocked. CPU eligibility
and actual argument-branch tests cover CUDA, HIP and Metal compile definitions. They cannot establish GPU numerical correctness. HIP-configured host
translation units compile; a complete linked runtime, other backends, parity,
cold cost, concurrent cancellation and performance remain qualification gates.
