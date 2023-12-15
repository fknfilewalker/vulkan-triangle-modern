cmake_minimum_required(VERSION 3.14)
project(vulkan-triangle-modern)

#================================#
# Dependencies                   #
#================================#
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VTM_EXTERNAL_FOLDER "Dependencies")
set(VTM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(VTM_EXTERNAL_DIR "${VTM_SOURCE_DIR}/external")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#================================#
# Dependencies                   #
#================================#
if(APPLE)
    # for moltenvk
    find_package(Vulkan REQUIRED)
    target_link_libraries(${LIB_RVK} PUBLIC Vulkan::Vulkan)
else()
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
        VK_NO_PROTOTYPES
    )
    set_target_properties(VulkanCppModule PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/vulkan")
endif()

if(WIN32)
    target_compile_definitions(VulkanCppModule PUBLIC 
        VK_USE_PLATFORM_WIN32_KHR 
        GLFW_EXPOSE_NATIVE_WIN32 
        NOMINMAX
    )
elseif(APPLE)
elseif(UNIX)
endif()

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory("${VTM_EXTERNAL_DIR}/glfw" EXCLUDE_FROM_ALL)
set_target_properties(glfw update_mappings PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/glfw")

#================================#
# Main Target                    #
#================================#
add_executable(${PROJECT_NAME} src/main.cpp src/shaders.h)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
add_dependencies(${PROJECT_NAME} glfw)

if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE VulkanCppModule)
    add_dependencies(${PROJECT_NAME} VulkanCppModule)
endif()

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