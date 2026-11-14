# Builds `muscriptor_notes` (notes.cpp), which runs MuScriptor through muscriptor.cpp, its C++/ggml port
# (https://github.com/DamRsn/muscriptor.cpp, MIT), for dev/muscriptor_eval.py. The tool itself does not use it.
#
#   git clone https://github.com/DamRsn/muscriptor.cpp
#   cmake -S dev/muscriptor_cpp -B build/muscriptor -G Ninja -DCMAKE_BUILD_TYPE=Release -DMUSCRIPTOR_CPP=<the clone>
#   cmake --build build/muscriptor
#
# Needs CMake 3.24+, Ninja, a C++23 compiler, on a Mac Xcode's Metal toolchain, and the network: configuring
# fetches ggml. The weights are a separate download (https://huggingface.co/DamRsn/muscriptor-gguf), under
# CC BY-NC 4.0 with the conditions of use of their authors, Mirelo and Kyutai.
cmake_minimum_required(VERSION 3.24)
project(muscriptor_notes LANGUAGES C CXX)

set(MUSCRIPTOR_CPP "" CACHE PATH "A checkout of https://github.com/DamRsn/muscriptor.cpp")
if(NOT EXISTS "${MUSCRIPTOR_CPP}/cpp/CMakeLists.txt")
    message(FATAL_ERROR "Set -DMUSCRIPTOR_CPP to a checkout of https://github.com/DamRsn/muscriptor.cpp")
endif()

add_subdirectory("${MUSCRIPTOR_CPP}/cpp" muscriptor)
add_executable(muscriptor_notes notes.cpp)
target_link_libraries(muscriptor_notes PRIVATE muscriptor_ggml)
muscriptor_add_metal_library(muscriptor_notes)
