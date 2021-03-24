cmake_minimum_required(VERSION 3.10)

# set the project name
project(Test)

#set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")


# add the executable
add_executable(Test main.cpp)

target_include_directories(Test PUBLIC
    "/opt/intel/openvino_2021/inference_engine/include/")

#find_package(InferenceEngine 2.1 REQUIRED)
find_library(INPUT_LIBRARYS inference_engine PATH /opt/intel/openvino_2021/inference_engine/lib/intel64/
/opt/intel/openvino_2021/inference_engine/external/tbb/lib/
/opt/intel/openvino_2021/deployment_tools/ngraph/lib/
)
target_link_libraries(Test ${INPUT_LIBRARYS})
#target_link_libraries(Test PUBLIC
#    "/opt/intel/openvino_2021/inference_engine/lib/intel64/libinference_engine.so"
#    "/opt/intel/openvino_2021/inference_engine/lib/intel64/libinference_engine.so"
#    "/opt/intel/openvino_2021/inference_engine/external/tbb/lib/libtbb.so"
#    "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so" )

