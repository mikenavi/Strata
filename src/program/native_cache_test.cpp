#include "strata/core/conversation_cache.hpp"
#include "strata/program/native_cache.hpp"
#include "strata/program/conv_cache.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace strata::core;
namespace nc = strata::program::native_cache;
using Chain = std::vector<ConversationCheckpoint>;
namespace {
int checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
std::vector<int32_t> tokens(int branch, int n) {
    std::vector<int32_t> ids;
    for (int i = 0; i < n; ++i) ids.push_back(i < 4 ? i : branch * 100 + i);
    return ids;
}
ConversationCheckpoint cp(int branch, int n) {
    ConversationCheckpoint c;
    c.ids = tokens(branch, n); c.used = n;
    c.gdn.assign(64, uint8_t(branch)); c.ple.assign(16, uint8_t(n));
    c.tails.assign(12, 3); c.dead.assign(8, 4); c.block_pos.assign(4, 5);
    return c;
}
int64_t best(const Chain& cs, const std::vector<int32_t>& prompt) {
    int64_t n = 0;
    for (const auto& c : cs) n = std::max(n, conversation_prefix(c, prompt, {}));
    return n;
}
SavedConversation park(Chain&& cs, int branch) {
    SavedConversation out; out.live = cp(branch, 16); out.checkpoints = std::move(cs);
    return out;
}
struct FailingCheckpoint : ConversationCheckpoint {
    static inline bool fail_copy = false;
    FailingCheckpoint() = default;
    FailingCheckpoint(const FailingCheckpoint& other) : ConversationCheckpoint(other) {
        if (fail_copy) throw std::bad_alloc();
    }
    FailingCheckpoint(FailingCheckpoint&&) noexcept = default;
    FailingCheckpoint& operator=(FailingCheckpoint&&) noexcept = default;
};
void equal_payload(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    check(a.ids == b.ids && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
        a.dead == b.dead && a.block_pos == b.block_pos, "all running-state payloads and token identities preserved");
}
}
int main() {
    constexpr uint64_t ample = 1 << 20, floor = uint64_t(8) << 30;
    {
        check(nc::room_above_floor({}, 0) == 0, "unknown memory fails closed");
        check(nc::room_above_floor(floor - 1, 0) == 0, "8 GiB floor enforced despite smaller configured floor");
        check(nc::room_above_floor(floor + 40, floor) == 40, "only room above floor admitted");
        check(nc::room_above_floor(floor + 40, floor + 100) == 0, "higher configured floor honored");
        check(nc::add(UINT64_MAX, 1) == UINT64_MAX, "byte addition saturates");
        Chain empty;
        check(!nc::admits(empty, UINT64_MAX, UINT64_MAX, UINT64_MAX), "overflow estimate rejected");
        auto consumed = tokens(1, 16), visible = consumed;
        visible.push_back(999); auto request = visible; request.push_back(1000);
        check(nc::lcp(consumed, request) == 16 && nc::lcp(visible, request) == 17,
            "unconsumed last emitted token cannot extend observed LCP");
        check(nc::observed_cut(4, 20, 8) == 8, "observed fork creates actual cut");
        check(nc::observed_cut(8, 20, 8) == -1, "already restored point is not saved again");
        check(nc::observed_cut(0, 20, 20) == 19, "full match capped at completed prefill extent");
        check(nc::observed_cut(0, 1, 1) == -1, "one-token input has no prefill snapshot");
    }
    {
        Chain main{cp(1,4), cp(1,8), cp(1,12)};
        std::array<Chain,2> slots;
        slots[0] = {main.back()}; // stock retains only the turn checkpoint
        auto fork = tokens(1,16); fork[8] = 999;
        check(best(slots[0], fork) == 0, "regression: deepest-only slot loses reusable earlier state");
        check(nc::handoff(slots[0], main, tokens(1,16), 16, 3, ample, ample) == 2,
            "main -> slot retains bounded chain");
        check(best(slots[0], fork) == 8, "retained middle checkpoint serves fork");
        equal_payload(slots[0][1], main[1]);
        check(slots[0][1].gdn.data() != main[1].gdn.data(), "donor and recipient own independent state");
        main[1].gdn[0] = 200;
        check(slots[0][1].gdn[0] == 1, "recipient unaffected by donor write");
        slots[1] = {cp(2,12)}; Chain b{cp(2,4),cp(2,8),cp(2,12)};
        nc::handoff(slots[1], b, tokens(2,16), 16, 3, ample, ample);
        for (int round = 0; round < 4; ++round) { // A/B/A/B, two independent active owners
            const int ix = round % 2, branch = ix + 1;
            main = {slots[ix].back()};
            nc::handoff(main, slots[ix], tokens(branch,16), 16, 3, ample, ample);
            auto next = tokens(branch,16); next[8] = 999;
            check(best(main, next) == 8 && main.size() == 3, "A/B/A/B chain returned intact");
            check(best(slots[1-ix], next) == 4, "other slot matches shared root only");
            check(slots[ix][1].gdn[0] == branch, "other admissions never mutate active donor state");
        }
        ConversationCache cache(ample, 2);
        check(cache.put(park(std::move(slots[0]),1)), "park A");
        check(cache.put(park(std::move(slots[1]),2)), "park B");
        auto match = cache.best(fork, {}, true);
        check(match.tokens == 8 && cache.borrows(match), "parked chain middle point available for borrow");
        const auto* donor = cache.find(match.serial);
        main = {donor->checkpoints[1]};
        nc::handoff(main, donor->checkpoints, fork, 8, 3, ample, ample);
        check(main.size() == 2 && cache.size() == 2, "borrow retains bounded prefix and both parked owners");
        main.back().gdn[0] = 99;
        check(cache.find(match.serial)->checkpoints[1].gdn[0] == 1, "borrow cannot corrupt parked donor");
        check(cache.observed_text_lcp(fork,true) == 8, "parked consumed metadata yields actual LCP");
        check(cache.observed_text_lcp(fork,false) == 0, "steering mismatch excluded from observation");
        auto restored = cache.take(match.index);
        check(restored.checkpoints.size() == 3 && restored.checkpoints[1].native_soft,
            "take transfers full chain and eviction metadata");
        check(cache.bytes() <= ample && cache.size() == 1, "park byte/count accounting remains bounded");
    }
    {
        Chain src{cp(1,4),cp(1,8),cp(1,12)}, dst{src.back()};
        check(nc::handoff(dst,src,tokens(1,16),16,3,ample,0) == 0 && dst.size() == 1,
            "RAM saturation preserves mandatory state without optional copy");
        check(nc::handoff(dst,src,tokens(1,16),16,3,nc::bytes(dst),ample) == 0,
            "byte saturation prevents optional chain growth");
        check(nc::handoff(dst,src,tokens(1,16),16,1,ample,ample) == 0, "count cap honored");
        auto bad = cp(1,6); bad.imgs.push_back({0,7}); src.push_back(bad);
        bad = cp(1,7); bad.stage_parts.emplace_back(); src.push_back(bad);
        bad = cp(2,9); src.push_back(bad);
        nc::handoff(dst,src,tokens(1,16),8,6,ample,ample);
        check(dst.size() == 3 && dst[0].ids.size() == 4 && dst[1].ids.size() == 8,
            "image/stage/off-branch/over-ceiling candidates excluded");
        dst[0].pinned = true;
        check(nc::trim_soft(dst,0) == 1 && dst.size() == 2 && dst.front().pinned,
            "pressure evicts optional state, preserves explicit pins and mandatory baseline");
        Chain soft{cp(1,4),cp(1,8)};
        soft[0].native_soft = soft[1].native_soft = true;
        check(nc::trim_soft(soft,0) == 2 && soft.empty(), "even soft root evictable, no automatic hard pin");
    }
    {
        // Two slots share physical RAM; neither receives an independent copy of
        // the initial headroom. Parked bytes are charged before both admissions.
        Chain src{cp(1,4),cp(1,8),cp(1,12)};
        std::array<Chain,2> slots;
        const uint64_t one_cost = src.front().bytes() + 2*sizeof(ConversationCheckpoint);
        uint64_t physical_available = floor + one_cost;
        const size_t before = nc::bytes(slots[0]);
        check(nc::handoff(slots[0],src,tokens(1,16),16,1,ample,
            nc::room_above_floor(physical_available, floor)) == 1, "first slot fits shared physical headroom");
        physical_available -= nc::bytes(slots[0]) - before;
        check(nc::handoff(slots[1],src,tokens(1,16),16,1,ample,
            nc::room_above_floor(physical_available, floor)) == 0, "second slot cannot spend first slot's headroom");
        check(physical_available >= floor, "two-slot retained bytes leave 8 GiB physical floor");
        ConversationCache tiny(1,2);
        check(!tiny.put(park(std::move(slots[0]),1)) && tiny.bytes() == 0,
            "parked budget refuses inherited chain when full snapshot cannot fit");
        auto image = park(Chain{cp(1,4),cp(1,8),cp(1,12)},1);
        image.live.imgs.push_back({5,42});
        ConversationCache images(ample,2);
        check(images.put(std::move(image)), "park image-bearing conversation for exclusion test");
        check(images.observed_text_lcp(tokens(1,20),true) == 0,
            "image-bearing parked history cannot supply text-only observation");
    }
    {
        // An interrupted prefill has exact earlier checkpoints, not state at the desired LCP.
        check(!nc::capture_ready(8,8,false,false), "in-flight cancellation cannot manufacture observed state");
        check(!nc::capture_ready(8,8,true,false), "successful return without exact state cannot save");
        check(!nc::capture_ready(7,8,true,true), "cancel before target does not save target");
        check(nc::capture_ready(8,8,true,true), "retry reaching completed target can save");
        Chain main{cp(1,4),cp(1,8),cp(1,12)};
        const auto* state = main[0].gdn.data();
        Chain yielded = std::move(main);
        nc::prune(yielded,tokens(1,6),6);
        check(main.empty() && yielded.size() == 1 && yielded[0].gdn.data() == state,
            "BYIELD moves exact compatible prefix without state duplication");
        Chain resumed = std::move(yielded);
        check(yielded.empty() && resumed[0].gdn.data() == state, "BYIELD retry preserves sole buffer owner");
        auto rewrite = tokens(1,16); rewrite[4] = 555;
        nc::prune(resumed,rewrite,8);
        check(resumed.size() == 1 && resumed[0].ids.size() == 4, "rewrite retains only valid backing prefix");
    }
    {
        FailingCheckpoint c;
        static_cast<ConversationCheckpoint&>(c) = cp(1,4);
        std::vector<FailingCheckpoint> donor; donor.push_back(std::move(c));
        std::vector<FailingCheckpoint> recipient;
        FailingCheckpoint::fail_copy = true;
        check(nc::handoff(recipient,donor,tokens(1,16),16,3,ample,ample) == 0,
            "optional allocation failure declines handoff");
        check(recipient.empty() && donor[0].ids == tokens(1,4) && donor[0].gdn[0] == 1,
            "allocation failure preserves donor identity/state and recipient invariants");
        FailingCheckpoint::fail_copy = false;
        check(nc::handoff(recipient,donor,tokens(1,16),16,3,ample,ample) == 1,
            "allocation failure retry succeeds without stranded ownership");
    }
    {
        std::mt19937 rng(20261025);
        for (int iteration = 0; iteration < 2000; ++iteration) {
            const int count = 1 + rng()%6, cap = 1 + rng()%6;
            const uint64_t limit = 256 + rng()%12000, room = rng()%12000;
            Chain src;
            for (int i = 1; i <= count; ++i) src.push_back(cp(1,i*4));
            Chain dst;
            const auto before = src.front().gdn;
            nc::handoff(dst,src,tokens(1,32),rng()%33,cap,limit,room);
            check(dst.size() <= (size_t)cap && nc::bytes(dst) <= limit,
                "randomized transfer respects count and actual capacity bytes");
            check(src.front().gdn == before, "randomized transfer never steals donor state");
            check(nc::bytes(dst) <= room, "sequential allocations remain inside sampled room");
            const int64_t read = rng()%32, observed = rng()%40, n = 1+rng()%40;
            const int64_t cut = nc::observed_cut(read,n,observed);
            check(cut == -1 || (cut > read && cut <= n-1 && cut <= observed), "randomized cut cannot overclaim state");
        }
    }
    std::printf("PASS native cache: %d checks; real checkpoint/cache types, CPU policy/ownership only\n", checks);
}
