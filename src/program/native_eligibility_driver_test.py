#!/usr/bin/env python3
"""Extract the real argument branch; test enabled/disabled arguments without starting a driver."""
from pathlib import Path
import sys
s=(Path(__file__).resolve().parent/'generate.cpp').read_text()
start=s.index('        else if (a == "--prompt-cache-native") {')
end=s.index('\n        else if (a == "--prompt-cache-tail")',start)
branch=s[start:end].replace('else if','if',1)
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'native-flag-test.cpp').write_text(r'''
#include "strata/program/native_cache.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
int parse(const char* value) {
    struct { int64_t prompt_cache_native_mib=0; } o;
    std::string a="--prompt-cache-native";
    auto next=[&](const char*) { return value; };
'''+branch+r'''
    return 0;
}
int main() {
    const bool supported=strata::program::native_cache::supported_backend(strata::program::native_cache::build_backend);
    if (parse("0")!=0 || parse("1024")!=(supported?0:1)) return 1;
    puts("PASS extracted actual argument branch: zero preserved, positive opt-in only CUDA/HIP");
}
''')
