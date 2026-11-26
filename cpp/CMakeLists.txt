cmake_minimum_required(VERSION 3.20)
project(lid_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

set(ONNXRUNTIME_ROOT "$ENV{HOME}/opt/onnxruntime" CACHE PATH "Diretorio do ONNX Runtime")

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
	IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
	INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_compile_options(-Wall -Wextra)
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

add_library(lid_core STATIC
	src/wav.cpp
	src/log_mel.cpp
	src/classifier.cpp
	src/util.cpp
)
target_include_directories(lid_core PUBLIC src)
target_link_libraries(lid_core PUBLIC onnxruntime)

add_executable(check_onnx src/check_onnx.cpp)
target_link_libraries(check_onnx PRIVATE lid_core)

add_executable(check_pipeline src/check_pipeline.cpp)
target_link_libraries(check_pipeline PRIVATE lid_core)

add_executable(lid src/lid.cpp)
target_link_libraries(lid PRIVATE lid_core)
