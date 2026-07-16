cmake_minimum_required(VERSION 3.28...3.30)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)
project(${_name} VERSION ${_version} LANGUAGES C CXX)

include(FetchContent)
include(compilerconfig)
include(defaults)
include(helpers)

option(BADDIECAM_DOWNLOAD_MODELS "Download pinned local AI models during configure" ON)
set(BADDIECAM_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models")
file(MAKE_DIRECTORY "${BADDIECAM_MODEL_DIR}")

function(baddiecam_download_model filename url sha256)
  set(destination "${BADDIECAM_MODEL_DIR}/${filename}")
  if(EXISTS "${destination}")
    file(SHA256 "${destination}" existing_hash)
    if("${existing_hash}" STREQUAL "${sha256}")
      message(STATUS "BaddieCam AI model verified: ${filename}")
      return()
    endif()
    file(REMOVE "${destination}")
  endif()
  if(NOT BADDIECAM_DOWNLOAD_MODELS)
    message(FATAL_ERROR "Missing ${filename}. Run scripts/DOWNLOAD_AI_MODELS.ps1 or configure with BADDIECAM_DOWNLOAD_MODELS=ON.")
  endif()
  message(STATUS "Downloading pinned BaddieCam AI model: ${filename}")
  file(DOWNLOAD "${url}" "${destination}"
       EXPECTED_HASH "SHA256=${sha256}"
       STATUS download_status
       SHOW_PROGRESS
       TLS_VERIFY ON)
  list(GET download_status 0 status_code)
  list(GET download_status 1 status_message)
  if(NOT status_code EQUAL 0)
    file(REMOVE "${destination}")
    message(FATAL_ERROR "Failed to download ${filename}: ${status_message}")
  endif()
endfunction()

if(WIN32)
  baddiecam_download_model(
    "face_detection_yunet_2023mar.onnx"
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
  baddiecam_download_model(
    "face_parsing_resnet18.onnx"
    "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
    "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f")
endif()

add_library(${CMAKE_PROJECT_NAME} MODULE)
find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

set(ORT_VERSION "1.24.4" CACHE STRING "ONNX Runtime DirectML NuGet version")
if(WIN32)
  FetchContent_Declare(
    onnxruntime_directml
    URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VERSION}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(onnxruntime_directml)
  set(ORT_ROOT "${onnxruntime_directml_SOURCE_DIR}")
  target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ORT_ROOT}/build/native/include")
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.lib"
  )
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BADDIECAM_WITH_ONNXRUNTIME=1)
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
  src/plugin-main.cpp
  src/beautycore-filter.cpp
  src/presets.cpp
  src/ai/image.cpp
  src/ai/mask-stabilizer.cpp
  src/ai/ort-session.cpp
  src/ai/yunet.cpp
  src/ai/face-parser.cpp
  src/ai/ai-engine.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE src)
set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

if(WIN32)
  foreach(_dll onnxruntime.dll DirectML.dll onnxruntime_providers_shared.dll)
    if(EXISTS "${ORT_ROOT}/runtimes/win-x64/native/${_dll}")
      add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ORT_ROOT}/runtimes/win-x64/native/${_dll}"
          "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/${_dll}")
    endif()
  endforeach()
endif()
