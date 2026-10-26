#!/usr/bin/env python3
"""Extract the actual driver save boundary, replacing only GPU/telemetry dependencies.
Output is a standalone CPU fault-injection program; no model or device is used.
"""
from pathlib import Path
import sys
source = Path(__file__).resolve().parent / "generate.cpp"
s = source.read_text()
start = s.index("        auto checkpoint_at = [&]")
end = s.index("\n        // STRATA_SPLIT_MTP_BATCH=1", start)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "extracted-checkpoint-at.inc").write_text(s[start:end] + "\n")
HEAD = r'''
#include "strata/core/conversation_cache.hpp"
#include "strata/program/native_cache.hpp"
#include "strata/program/conv_cache.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <string>
using ConvCheckpoint = strata::core::ConversationCheckpoint;
using Clock = std::chrono::steady_clock;
static int fail_at = -1, allocation_index = 0;
static bool injected = false, use_pool = false, use_threads = false, save_ok = true, sync_ok = true;
void* operator new(std::size_t n) { if (allocation_index++ == fail_at) { fail_at = -1; injected = true; throw std::bad_alloc(); } if (auto p = std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
namespace strata::core {
struct ConversationStateSizes { uint64_t gdn = 64, ple = 16, tail = 12, dead = 8, block_pos = 4; };
struct SessionState { int64_t qsa_alloc = 1; };
struct Geometry {};
struct OnDevice { explicit OnDevice(int) {} };
bool conversation_session_sizes(const Geometry&, const SessionState&, ConversationStateSizes&, std::string&) { return true; }
}
static bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState&, const strata::core::Geometry&) {
    c.gdn.resize(64); c.ple.resize(16); c.tails.resize(12); c.dead.resize(8); c.block_pos.resize(4); return save_ok;
}
using cudaError_t = int;
constexpr int cudaSuccess = 0;
int cudaDeviceSynchronize() { return sync_ok ? cudaSuccess : 1; }
const char* cudaGetErrorString(int) { return "mock"; }
bool ckpt_threads_env() { return use_threads; }
struct Pool { bool on() const { return use_pool; } ConvCheckpoint take() { return {}; } void give(ConvCheckpoint&&) {} template<class F> void erase_if(std::vector<ConvCheckpoint>& cs, F f) { std::erase_if(cs, f); } };
struct Threads { void run(size_t n, const std::function<void(size_t)>& f) { for (size_t i=0;i<n;++i) f(i); } };
struct Stage { int dev=0; strata::core::SessionState ss; };
int main() {
    struct { int prompt_cache=3; bool prompt_cache_spread=false; } o;
    std::string ckpt_why;
    std::vector<ConvCheckpoint> checks;
    uint64_t check_clock=0;
    int64_t tail_ckpt_len=-1;
    strata::core::SessionState ss;
    strata::core::Geometry g;
    const uint64_t native_chain_bytes=1024*1024;
    auto native_room=[] { return uint64_t(1024*1024); };
    bool ckpt_trace=false;
    Pool ckpt_pool;
    Threads ckpt_threads;
    std::vector<std::unique_ptr<Stage>> stages;
    std::vector<int64_t> cur{1,2,3,4,5,6,7,8,9,10};
    std::vector<strata::core::ConversationImageKey> req_imgs;
    auto imgs_below=[](const auto&, int64_t) { return std::vector<strata::core::ConversationImageKey>{}; };
    auto gpu_sync_ok=[] { return sync_ok; };
    ConvCheckpoint::PromptPlan seg_plan;
    struct { int ckpts=0; double sync_ms=0,save_ms=0,evict_ms=0; } req_times;
    bool multi_gpu=false;
'''
TAIL = r'''
    int assertions = 0, failure_cases = 0;
    auto require = [&](bool ok, const char* why) {
        ++assertions;
        if (!ok) { std::fprintf(stderr, "FAIL actual checkpoint_at: %s\n", why); std::exit(1); }
    };
    auto checkpoint = [](int n, bool pin) {
        ConvCheckpoint c;
        for (int i=1;i<=n;++i) c.ids.push_back(i);
        c.gdn.assign(64,11); c.ple.assign(16,12); c.tails.assign(12,13);
        c.dead.assign(8,14); c.block_pos.assign(4,15); c.used=n; c.pinned=pin;
        return c;
    };
    auto same = [](const auto& a, const auto& b) {
        if (a.size()!=b.size()) return false;
        for (size_t i=0;i<a.size();++i)
            if (a[i].ids!=b[i].ids || a[i].gdn!=b[i].gdn || a[i].ple!=b[i].ple ||
                a[i].tails!=b[i].tails || a[i].dead!=b[i].dead || a[i].block_pos!=b[i].block_pos ||
                a[i].used!=b[i].used || a[i].pinned!=b[i].pinned || a[i].native_soft!=b[i].native_soft)
                return false;
        return true;
    };
    // Two untouched external owners stand for active and parked donors. The actual
    // extracted save boundary must never consume either while constructing a snapshot.
    std::vector<ConvCheckpoint> donor_a{checkpoint(3,true)}, donor_b{checkpoint(5,false)};
    const auto original_a=donor_a, original_b=donor_b;
    for (bool pool : {false,true}) for (bool threads : {false,true})
    for (int setup : {0,1,2}) { // empty, full with evictable state, all explicit pins
        use_pool=pool; use_threads=threads;
        auto reset = [&] {
            fail_at=-1; injected=false; checks.clear(); checks.shrink_to_fit();
            if (setup) checks={checkpoint(2,true),checkpoint(4,true),checkpoint(6,setup==2)};
            check_clock=10; tail_ckpt_len=-1; req_times={};
        };
        reset();
        allocation_index=0;
        require(checkpoint_at(8,nullptr,false,true), "successful optional capture returns true");
        const int count=allocation_index;
        require(checks.size()<=size_t(o.prompt_cache), "successful capture obeys count");
        require(setup==2 ? checks.back().ids.size()==6 : checks.back().ids.size()==8,
            "full all-pinned chain skips optional publication; other cases capture");
        for (int fail=0;fail<count;++fail) {
            reset();
            const auto before=checks;
            std::vector<const uint8_t*> owners;
            for (const auto& c:checks) owners.push_back(c.gdn.data());
            allocation_index=0; fail_at=fail;
            bool result=false;
            try { result=checkpoint_at(8,nullptr,false,true); }
            catch (const std::bad_alloc&) { fail_at=-1; require(false,"optional allocation failure escaped"); }
            fail_at=-1;
            require(injected && result,"every injected allocation is a recoverable skip");
            require(same(checks,before) && checks.size()<=size_t(o.prompt_cache),
                    "failed snapshot/publication preserves full prior chain and count");
            require(check_clock==10 && tail_ckpt_len==-1,"failed capture cannot commit clock/tail metadata");
            for (size_t i=0;i<checks.size();++i)
                require(checks[i].gdn.data()==owners[i],"failed capture preserves mandatory buffer owner");
            require(same(donor_a,original_a)&&same(donor_b,original_b),"both donor owners survive");
            ++failure_cases;
            require(checkpoint_at(8,nullptr,false,true),"retry after memory failure succeeds or safely skips");
            require(checks.size()<=size_t(o.prompt_cache),"retry obeys count");
        }
    }
    use_pool=false; use_threads=false; checks.clear();
    fail_at=0; allocation_index=0;
    bool mandatory_threw=false;
    try { checkpoint_at(8,nullptr,false,false); }
    catch (const std::bad_alloc&) { mandatory_threw=true; }
    fail_at=-1; require(mandatory_threw,"mandatory failures retain upstream exception semantics");
    save_ok=false; require(!checkpoint_at(8,nullptr,false,true),"GPU save error is not a memory skip");
    save_ok=true; sync_ok=false;
    require(!checkpoint_at(8,nullptr,false,true),"GPU synchronization error is not swallowed");
    sync_ok=true;
    std::printf("PASS actual optional-save boundary: %d allocation failures, %d assertions; snapshot/GPU operations mocked\n",
                failure_cases,assertions);
}
'''
(out / "native-optional-save-test.cpp").write_text(HEAD + '\n#include "extracted-checkpoint-at.inc"\n' + TAIL)
print("Extracted actual driver checkpoint_at for exhaustive allocation failure test")
