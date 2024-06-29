# Specifies the minimum version of CMake required to build your project
cmake_minimum_required(VERSION 3.10) 
# Defines the name of the project. This will set some default variables and configurations based on the project name
project(FaceDetection)

# Finds and loads the OpenCV package, making its settings and variables available
# REQUIRED means that the configuration will fail if OpenCV is not found
find_package(OpenCV REQUIRED)
# Adds the include directories for OpenCV to the compiler's search path
# This allows your project to find and include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})
# Defines an executable target called "FaceDetection" and specifies the source file "main.cpp" for the target
add_executable(FaceDetection main.cpp) 
# Links the OpenCV libraries to the "FaceDetection" target
# This ensures that the necessary OpenCV libraries are linked during the build process
target_link_libraries(FaceDetection ${OpenCV_LIBS})
