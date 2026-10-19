cmake_minimum_required(VERSION 3.30)
# vulkan.cppm uses `import std;`, which is still behind an experimental gate (the UUID changes with CMake versions)
# see Help/dev/experimental.rst in the CMake source tree of each version
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.5)
    message(WARNING "Unknown `import std` UUID for CMake ${CMAKE_VERSION}, trying the one of 4.4")
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 4.4)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 4.3)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0.3)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31.8)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
else()
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
endif()
# brew LLVM ships libc++.modules.json in lib/c++, where `clang -print-file-name` does not look
if(APPLE AND CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    get_filename_component(_llvm_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    if(EXISTS "${_llvm_bin_dir}/../lib/c++/libc++.modules.json")
        set(CMAKE_CXX_STDLIB_MODULES_JSON "${_llvm_bin_dir}/../lib/c++/libc++.modules.json")
    endif()
endif()
project(vulkan-triangle-modern)

#================================#
# Dependencies                   #
#================================#
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
	set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -Wno-deprecated-declarations")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
endif ()

set(VTM_EXTERNAL_FOLDER "Dependencies")
set(VTM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(VTM_EXTERNAL_DIR "${VTM_SOURCE_DIR}/external")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#================================#
# Dependencies                   #
#================================#
# set up vulkan C++ module target
add_library(VulkanCppModule)
target_sources(VulkanCppModule PUBLIC FILE_SET CXX_MODULES
    FILES "${VTM_EXTERNAL_DIR}/vulkan-headers/include/vulkan/vulkan.cppm"
)
target_include_directories(VulkanCppModule PUBLIC "${VTM_EXTERNAL_DIR}/vulkan-headers/include")
target_compile_definitions(VulkanCppModule PUBLIC 
    VULKAN_HPP_NO_SMART_HANDLE
    VULKAN_HPP_NO_SPACESHIP_OPERATOR
    VULKAN_HPP_NO_TO_STRING
    VULKAN_HPP_NO_STD_MODULE
    VK_NO_PROTOTYPES
)
set_target_properties(VulkanCppModule PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/vulkan")

if(WIN32)
    target_compile_definitions(VulkanCppModule PUBLIC
        NOMINMAX
    )
elseif(APPLE)
    find_package(Vulkan REQUIRED)
    target_link_libraries(VulkanCppModule PUBLIC "Vulkan::Vulkan")
elseif(UNIX)
endif()

set(SDL_DISABLE_UNINSTALL ON)
set(SDL_TEST_LIBRARY OFF)
set(SDL_SHARED OFF)
set(SDL_STATIC ON)
# # subsystems
set(SDL_AUDIO OFF)
set(SDL_RENDER OFF)
set(SDL_CAMERA OFF)
set(SDL_JOYSTICK OFF)
set(SDL_HAPTIC OFF)
set(SDL_HIDAPI ON)
set(SDL_SENSOR OFF)
set(SDL_DIALOG OFF)
set(SDL_DIRECTX OFF)
set(SDL_OPENGLES OFF)
set(SDL_VULKAN ON)
set(SDL_OFFSCREEN OFF)
add_subdirectory("${VTM_EXTERNAL_DIR}/sdl" EXCLUDE_FROM_ALL)
target_compile_definitions(SDL3-static PUBLIC SDL_MAIN_HANDLED)
set_target_properties(SDL3-static PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/sdl")

#================================#
# Main Target                    #
#================================#
add_executable(${PROJECT_NAME} src/main.cpp src/shaders.h)
target_link_libraries(${PROJECT_NAME} PRIVATE VulkanCppModule SDL3-static)
add_dependencies(${PROJECT_NAME} VulkanCppModule SDL3-static)

install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Debug DESTINATION "debug")
install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release DESTINATION "release")

#================================#
# IDE specific setup              #
#================================#
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
elseif(CMAKE_GENERATOR MATCHES "Xcode")
    set(CMAKE_XCODE_GENERATE_SCHEME ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_GENERATE_SCHEME ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY XCODE_STARTUP_PROJECT ${PROJECT_NAME})
endif()
