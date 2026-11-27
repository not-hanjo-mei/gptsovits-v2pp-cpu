cmake_minimum_required(VERSION 3.20)
project(potato_tts_host CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)  # stage 3: kernels need -O3; -O0 cost 5-10x
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT POTATO_FREE_ONLY)
  set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ort)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ORT_DIR}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/include)
endif()

# POTATO_FREE_ONLY: build tts_host with the ORT engine path compiled out
# (main.cpp guards on the macro of the same name; packed.h/vits_packed.h
# are bypassed for packed_core.h) and no onnxruntime linkage. Default
# OFF — the shipping dual-engine build is unchanged. See ANDROID-SPIKE.md.
option(POTATO_FREE_ONLY "tts_host: free engine only, no ONNX Runtime" OFF)

# Hand-written AVX2 conv tile core (x86_64 only): vits_kernels.cpp
# routes eligible chunks to it when POTATO_HAVE_ASM_TILE is set.
# POTATO_NO_ASM=1 at runtime disables it without a rebuild. This
# block must precede the add_executable calls that use the var.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    enable_language(ASM)
    set(POTATO_ASM_SRC src/conv16_asm.S)
    add_compile_definitions(POTATO_HAVE_ASM_TILE=1)
else()
    set(POTATO_ASM_SRC "")
endif()

add_executable(tts_host src/main.cpp src/g2p.cpp src/vits_kernels.cpp src/free_engine.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
if(POTATO_FREE_ONLY)
  target_compile_definitions(tts_host PRIVATE POTATO_FREE_ONLY=1)
  if(NOT ANDROID)
    # bionic has pthread inside libc; there is no separate -lpthread.
    target_link_libraries(tts_host PRIVATE pthread)
  endif()
else()
  target_link_libraries(tts_host PRIVATE onnxruntime)
  set_target_properties(tts_host PROPERTIES
    BUILD_RPATH ${ORT_DIR}/lib
    INSTALL_RPATH ${ORT_DIR}/lib)
endif()

if(NOT POTATO_FREE_ONLY)
  add_executable(tf_check src/tf_check.cpp src/avx2_kernels.cpp)
  target_link_libraries(tf_check PRIVATE onnxruntime)

  add_executable(vits_check src/vits_check.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
  target_link_libraries(vits_check PRIVATE onnxruntime)

  add_executable(kbench src/kbench.cpp src/avx2_kernels.cpp)
  target_link_libraries(kbench PRIVATE onnxruntime pthread)
endif()

# GEMV tier micro-bench (packed_core.h only, no ORT): builds for the
# NDK target too, unlike kbench whose ORT import is x86-only.
add_executable(gemv_bench src/gemv_bench.cpp src/avx2_kernels.cpp)
if(NOT ANDROID)
  target_link_libraries(gemv_bench PRIVATE pthread)
endif()
target_include_directories(gemv_bench PRIVATE src)

# Packed-conv tile micro-bench (vits_kernels only, no ORT).
add_executable(vits_bench src/vits_bench.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
if(NOT ANDROID)
  target_link_libraries(vits_bench PRIVATE pthread)
endif()
target_include_directories(vits_bench PRIVATE src)

# ORT-free engine (ORT-FREE.md): GPT-side replacement, phase 1.
# Deliberately NOT linked against onnxruntime -- the point of the
# exercise. Parallel existence only; shipping defaults unchanged.
# Phase 2: vits_kernels.cpp joins for the packed conv kernels; its
# declarations moved to the ORT-free vits_kernels.h, so still no ORT
# headers or library in these binaries (verify with ldd).
add_executable(tf_check_free src/tf_check_free.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(tf_check_free PRIVATE pthread)
target_include_directories(tf_check_free PRIVATE src)

# ORT-free vits validation (phase 2): vits_check's protocol on the free
# engine. Also deliberately ORT-free; cross-checks against the ORT path
# through vits_check --dump files.
add_executable(vits_check_free src/vits_check_free.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(vits_check_free PRIVATE pthread)
target_include_directories(vits_check_free PRIVATE src)

# HuBERT gate for runtime reference extraction: engine output vs
# the ORT run and the shipped refpack tensor (see
# src/hubert_check.cpp).
add_executable(hubert_check src/hubert_check.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(hubert_check PRIVATE pthread)
target_include_directories(hubert_check PRIVATE src)

add_executable(roberta_check src/roberta_check.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(roberta_check PRIVATE pthread)
target_include_directories(roberta_check PRIVATE src)

add_executable(graph_check src/graph_check.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(graph_check PRIVATE pthread)
target_include_directories(graph_check PRIVATE src)

# Compile-only probe for surveying new graphs (reference
# extraction models) against the engine's op coverage.
add_executable(plan_probe src/plan_probe.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(plan_probe PRIVATE pthread)
target_include_directories(plan_probe PRIVATE src)

# Streaming validation: chunked VITS slicing vs one-shot on the
# noiseless fp32 graph (see src/vits_slice_check.cpp).
add_executable(vits_slice_check src/vits_slice_check.cpp src/free_engine.cpp src/vits_kernels.cpp src/avx2_kernels.cpp ${POTATO_ASM_SRC})
target_link_libraries(vits_slice_check PRIVATE pthread)
target_include_directories(vits_slice_check PRIVATE src)

add_executable(test_softclip src/test_softclip.cpp)
if(NOT POTATO_FREE_ONLY)
  set_target_properties(tf_check PROPERTIES
    BUILD_RPATH ${ORT_DIR}/lib
    INSTALL_RPATH ${ORT_DIR}/lib)
endif()

# M2: the AVX2 kernel TU is the only file compiled for the AVX2+FMA
# baseline; everything else stays generic so the scalar fallback paths
# keep their exact numerics (the packed gates depend on them).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(src/avx2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()


