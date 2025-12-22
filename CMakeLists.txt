cmake_minimum_required(VERSION 3.28)
project(vulkan-triangle-modern)

#================================#
# Dependencies                   #
#================================#
set(CMAKE_CXX_STANDARD 20)
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
set_target_properties(SDL_uclibc PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/sdl")

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
