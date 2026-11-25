cmake_minimum_required(VERSION 3.20)
project(lid_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "$ENV{HOME}/opt/onnxruntime" CACHE PATH "Diretorio do ONNX Runtime")

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
	IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
	INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_executable(check_onnx src/main.cpp)
target_link_libraries(check_onnx PRIVATE onnxruntime)
target_compile_options(check_onnx PRIVATE -Wall -Wextra)
set_target_properties(check_onnx PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
