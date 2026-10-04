cmake_minimum_required(VERSION 3.21)
project(strata_vulkan_s2_gemv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLANG glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

add_executable(s2_gemv_vk main.cpp)
target_link_libraries(s2_gemv_vk PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(s2_gemv_vk PRIVATE /W4)
endif()

# the shaders are compiled to SPIR-V next to the executable, which loads them at start-up
add_custom_command(
  TARGET s2_gemv_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv.comp
          -o $<TARGET_FILE_DIR:s2_gemv_vk>/s2_gemv.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv.comp
  COMMENT "glslangValidator: s2_gemv.comp -> s2_gemv.spv")
add_custom_command(
  TARGET s2_gemv_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv_wg.comp
          -o $<TARGET_FILE_DIR:s2_gemv_vk>/s2_gemv_wg.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv_wg.comp
  COMMENT "glslangValidator: s2_gemv_wg.comp -> s2_gemv_wg.spv")

# the BF16 GEMV port (P2.S5): warp-per-row and split modes in one shader, host mirrors bf16_gemv_parity.cpp
add_executable(bf16_gemv_vk bf16_main.cpp)
target_link_libraries(bf16_gemv_vk PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(bf16_gemv_vk PRIVATE /W4)
endif()
add_custom_command(
  TARGET bf16_gemv_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/bf16_gemv.comp
          -o $<TARGET_FILE_DIR:bf16_gemv_vk>/bf16_gemv.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bf16_gemv.comp
  COMMENT "glslangValidator: bf16_gemv.comp -> bf16_gemv.spv")

# the Q8_0-activation GEMV port (round 186): host mirrors s2_gemv_q8_parity.cpp; check 2 runs the ALREADY-PORTED
# s2_gemv shader as the fp16-activation rival, so this target needs both .spv files next to its exe.
add_executable(s2_gemv_q8_vk s2_gemv_q8_main.cpp)
target_link_libraries(s2_gemv_q8_vk PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(s2_gemv_q8_vk PRIVATE /W4)
endif()
add_custom_command(
  TARGET s2_gemv_q8_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv_q8.comp
          -o $<TARGET_FILE_DIR:s2_gemv_q8_vk>/s2_gemv_q8.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv_q8.comp
  COMMENT "glslangValidator: s2_gemv_q8.comp -> s2_gemv_q8.spv")
add_custom_command(
  TARGET s2_gemv_q8_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv.comp
          -o $<TARGET_FILE_DIR:s2_gemv_q8_vk>/s2_gemv.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/s2_gemv.comp
  COMMENT "glslangValidator: s2_gemv.comp -> s2_gemv.spv (fp16 rival)")

# the layer-glue port (kernel 4): gdn_gate/silu/scale/f32->f16/rms_norm_weighted/embedding_gather in one
# mode-switched shader; host mirrors elementwise_parity.cpp and needs shaderFloat64 for silu's contract.
add_executable(elementwise_vk elementwise_main.cpp)
target_link_libraries(elementwise_vk PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(elementwise_vk PRIVATE /W4)
endif()
add_custom_command(
  TARGET elementwise_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/elementwise.comp
          -o $<TARGET_FILE_DIR:elementwise_vk>/elementwise.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/elementwise.comp
  COMMENT "glslangValidator: elementwise.comp -> elementwise.spv")

# the activation quantizer port (kernel 5): Q8_0/Q8_K quantize+dequant + f16 table modes in one shader;
# host mirrors quantize_act_parity.cpp and needs shaderFloat64 for the double-division-before-rint contract.
add_executable(quantize_act_vk quantize_act_main.cpp)
target_link_libraries(quantize_act_vk PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(quantize_act_vk PRIVATE /W4)
endif()
add_custom_command(
  TARGET quantize_act_vk POST_BUILD
  COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/quantize_act.comp
          -o $<TARGET_FILE_DIR:quantize_act_vk>/quantize_act.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/quantize_act.comp
  COMMENT "glslangValidator: quantize_act.comp -> quantize_act.spv")

