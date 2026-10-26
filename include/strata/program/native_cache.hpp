// Bounded native checkpoint retention. This policy owns no KV and never mounts a
// state: callers must first copy/restore the same branch's KV through ceiling.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <vector>
#include <memory>
#include <type_traits>
#include "strata/program/conv_cache.hpp"

namespace strata::program::native_cache {
// This header is used by the shared CUDA/HIP/Metal driver. SYCL has a separate overlay.
enum class Backend { cuda, hip, metal, other };
#if defined(STRATA_GPU_METAL)
inline constexpr Backend build_backend = Backend::metal;
#elif defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
inline constexpr Backend build_backend = Backend::hip;
#else
inline constexpr Backend build_backend = Backend::cuda;
#endif
inline constexpr bool supported_backend(Backend backend) {
    return backend == Backend::cuda || backend == Backend::hip;
}
inline constexpr bool text_eligible(bool enabled, bool multi_gpu, bool has_images,
                                    Backend backend = build_backend) {
    return enabled && supported_backend(backend) && !multi_gpu && !has_images;
}
inline uint64_t parking_floor(uint64_t configured, bool enabled, bool multi_gpu, bool has_images,
                              Backend backend = build_backend) {
    return text_eligible(enabled, multi_gpu, has_images, backend)
        ? std::max<uint64_t>(uint64_t(8192) << 20, configured) : configured;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}
template<class Checkpoint>
uint64_t bytes(const std::vector<Checkpoint>& chain) {
    uint64_t n = chain.capacity() * sizeof(Checkpoint);
    for (const auto& c : chain) n = add(n, c.bytes());
    return n;
}
template<class A, class B>
int64_t lcp(const std::vector<A>& a, const std::vector<B>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return (int64_t) n;
}
inline uint64_t room_above_floor(std::optional<uint64_t> available, uint64_t configured_floor) {
    const uint64_t floor = std::max<uint64_t>(uint64_t(8192) << 20, configured_floor);
    return available && *available > floor ? *available - floor : 0;
}
// The last prompt token belongs to the decode window: at most prompt-1 tokens
// can be captured by prefill. Never rewind merely to make a checkpoint.
inline int64_t observed_cut(int64_t resume, int64_t prompt, int64_t observed) {
    if (prompt <= 1 || resume < 0) return -1;
    const int64_t at = std::min(observed, prompt - 1);
    return at > resume && at > 0 ? at : -1;
}
inline bool capture_ready(int64_t at, int64_t target, bool read_ok, bool exact) {
    return target > 0 && at == target && read_ok && exact;
}
template<class Checkpoint, class Token>
bool compatible(const Checkpoint& c, const std::vector<Token>& prefix, int64_t ceiling) {
    return ceiling > 0 && !c.ids.empty() && c.imgs.empty() && c.stage_parts.empty() &&
        c.ids.size() <= (uint64_t) ceiling && c.ids.size() <= prefix.size() &&
        std::equal(c.ids.begin(), c.ids.end(), prefix.begin());
}
template<class Checkpoint, class Token>
void prune(std::vector<Checkpoint>& chain, const std::vector<Token>& prefix, int64_t ceiling) {
    chain.erase(std::remove_if(chain.begin(), chain.end(), [&](const auto& c) {
        return !compatible(c, prefix, ceiling);
    }), chain.end());
}
// Existing ordinary checkpoints/pins retain their contract. Only additions made
// by this feature are reclaimable here, oldest first (even a soft root).
template<class Checkpoint>
size_t trim_soft(std::vector<Checkpoint>& chain, uint64_t ceiling) {
    size_t removed = 0;
    while (bytes(chain) > ceiling) {
        auto victim = chain.end();
        for (auto it = chain.begin(); it != chain.end(); ++it)
            if (it->native_soft && !it->pinned && (victim == chain.end() || it->used < victim->used)) victim = it;
        if (victim == chain.end()) break;
        chain.erase(victim); // free payloads; do not return optional bytes to a pooling allocator
        ++removed;
    }
    return removed;
}
template<class Checkpoint>
bool admits(const std::vector<Checkpoint>& dst, uint64_t payload, uint64_t ceiling, uint64_t room) {
    // Include a conservative transient directory during vector growth. Payloads
    // already in dst and all other slots/parked entries are already in MemAvailable.
    const uint64_t transient = add(payload, (uint64_t(dst.size()) + 1) * 2 * sizeof(Checkpoint));
    return transient != UINT64_MAX && transient <= room && add(bytes(dst), transient) <= ceiling;
}
// Prepare all potentially allocating work before publication. A failed reserve,
// metadata allocation or snapshot copy leaves every old checkpoint untouched.
// After reserve, nothrow moves/erase commit count eviction without another allocation.
template<class Checkpoint>
bool publish_optional(std::vector<Checkpoint>& chain, Checkpoint&& candidate,
                      int cap, uint64_t byte_limit, uint64_t room, bool spread, int64_t tail_len) {
    static_assert(std::is_nothrow_move_constructible_v<Checkpoint> &&
                  std::is_nothrow_move_assignable_v<Checkpoint>);
    if (cap <= 0 || chain.size() > (size_t) cap || candidate.pinned ||
        !admits(chain, candidate.bytes(), byte_limit, room)) return false;
    const size_t n = chain.size() + 1;
    size_t victim = n;
    if (n > (size_t) cap) {
        std::vector<uint64_t> stamps; stamps.reserve(n);
        std::vector<int64_t> lens; lens.reserve(n);
        std::unique_ptr<bool[]> tails(new bool[n]), pins(new bool[n]);
        for (size_t i = 0; i < n; ++i) {
            const auto& c = i < chain.size() ? chain[i] : candidate;
            stamps.push_back(c.used); lens.push_back((int64_t) c.ids.size());
            tails[i] = tail_len >= 0 && (int64_t) c.ids.size() == tail_len;
            pins[i] = c.pinned;
        }
        victim = spread
            ? conv_cache::spread_victim(lens.data(), stamps.data(), n, cap, tails.get(), pins.get())
            : conv_cache::eviction_victim(stamps.data(), n, cap, tails.get(), pins.get());
        // Stock's all-pinned fallback is not permission for optional work to drop a pin.
        if (victim >= chain.size() || pins[victim]) return false;
    }
    chain.reserve(n); // last throwing operation; all existing payload owners remain in place on failure
    chain.push_back(std::move(candidate));
    if (victim < n) chain.erase(chain.begin() + (std::ptrdiff_t) victim);
    return true;
}
// `dst` already holds the mandatory selected checkpoint. Independent copies
// keep an active donor valid when the recipient branches/cancels. Source remains
// untouched on allocation failure. Count and byte limits apply before each copy.
template<class Checkpoint, class Token>
size_t handoff(std::vector<Checkpoint>& dst, const std::vector<Checkpoint>& src,
               const std::vector<Token>& prefix, int64_t ceiling, int cap,
               uint64_t byte_limit, uint64_t room) {
    if (&dst == &src || cap <= 0 || ceiling <= 0) return 0;
    size_t added = 0;
    auto take = [&](const Checkpoint& c) {
        if (dst.size() >= (size_t) cap || !compatible(c, prefix, ceiling)) return;
        if (std::any_of(dst.begin(), dst.end(), [&](const auto& d) { return d.ids.size() == c.ids.size(); })) return;
        const uint64_t cost = add(c.bytes(), (uint64_t(dst.size()) + 1) * 2 * sizeof(Checkpoint));
        if (!admits(dst, c.bytes(), byte_limit, room)) return;
        try { dst.push_back(c); }
        catch (const std::bad_alloc&) { return; }
        // A copied explicit pin stays explicit. No automatic pin is introduced.
        dst.back().native_soft = true;
        room -= cost;
        ++added;
    };
    // Preserve the root when it fits, then the deepest checkpoints. `src` is the
    // existing sorted chain; stock LRU/spread eviction remains authoritative.
    if (!src.empty()) take(src.front());
    for (auto it = src.rbegin(); it != src.rend(); ++it) take(*it);
    std::sort(dst.begin(), dst.end(), [](const auto& a, const auto& b) { return a.ids.size() < b.ids.size(); });
    return added;
}
} // namespace strata::program::native_cache
