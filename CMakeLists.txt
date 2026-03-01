cmake_minimum_required(VERSION 3.16)

project(VulkanRenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)

# Source File Management - Simplified
# Note: Using GLOB is okay for small projects, but listing them is safer.
file(GLOB_RECURSE ALL_SOURCES
        "src/*.cpp"
        "src/*.h"
        "include/*.h"
        "include/*.hpp"
)

# Create the Executable
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

# Include Paths - Using PUBLIC/PRIVATE correctly
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linking
target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
)

message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")
