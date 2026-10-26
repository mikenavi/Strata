#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/program/native_cache.hpp"
#include <cstdio>
#include <cstdlib>
using namespace strata::core;
namespace nc = strata::program::native_cache;
int main() {
    int assertions=0;
    auto check = [&](bool ok, const char* why) {
        ++assertions;
        if (!ok) { std::fprintf(stderr,"FAIL eligibility: %s\n",why); std::exit(1); }
    };
#if defined(STRATA_GPU_METAL)
    check(nc::build_backend==nc::Backend::metal,"Metal shared driver selects unsupported backend");
#elif defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    check(nc::build_backend==nc::Backend::hip,"HIP shared driver selects supported backend");
#else
    check(nc::build_backend==nc::Backend::cuda,"default shared driver selects CUDA backend");
#endif
    constexpr uint64_t mib=uint64_t(1)<<20;
    for (auto backend : {nc::Backend::cuda,nc::Backend::hip,nc::Backend::metal,nc::Backend::other})
    for (bool enabled : {false,true}) for (bool split : {false,true}) for (bool image : {false,true}) {
        const bool expected=enabled&&!split&&!image&&(backend==nc::Backend::cuda||backend==nc::Backend::hip);
        check(nc::text_eligible(enabled,split,image,backend)==expected,"backend/text/image/flag matrix");
        check(nc::parking_floor(4096*mib,enabled,split,image,backend)==(expected?8192*mib:4096*mib),
              "outgoing parking has exactly the same supported-state eligibility");
        check(nc::parking_floor(12000*mib,enabled,split,image,backend)==12000*mib,
              "candidate cannot lower configured floor");
        ConversationCache parked(4096,2);
        for (int branch : {1,2}) {
            SavedConversation c; c.live.ids={branch}; c.live.gdn.assign(16,uint8_t(branch));
            check(parked.put(std::move(c)),"prepare two existing parked owners");
        }
        const auto prior_bytes=parked.bytes();
        // Same admission predicate and eviction-loop condition used by park_current_body.
        // Entries are tiny: even after eviction the mocked 6 GiB reading stays below 8 GiB.
        auto admit = [&] { return conversation_memory_admit(6144*mib,256*mib,
            nc::parking_floor(4096*mib,enabled,split,image,backend)); };
        while (!admit() && parked.size()>0 && parked.evict_oldest()) {}
        if (!expected) check(admit()&&parked.size()==2&&parked.bytes()==prior_bytes,
            "image/unsupported/disabled path keeps stock admission and both parked owners");
        else check(!admit()&&parked.size()==0,"supported text path enforces elevated floor");
    }
    check(!nc::text_eligible(true,false,true),"image exclusion also uses actual compiled backend");
    check(nc::text_eligible(true,false,false)==nc::supported_backend(nc::build_backend),
          "actual compiled backend gates text path");
    std::printf("PASS eligibility: %d assertions, compiled backend=%d\n",assertions,int(nc::build_backend));
}
