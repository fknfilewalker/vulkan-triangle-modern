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

#================================#
# Dependencies                   #
#================================#
#find_package(Vulkan REQUIRED)
# set up Vulkan C++ module
add_library(VulkanCppModule)
target_sources(VulkanCppModule PUBLIC
    FILE_SET CXX_MODULES
    FILES ${VTM_EXTERNAL_DIR}/vulkan-headers/include/vulkan/vulkan.cppm
)
target_include_directories(VulkanCppModule PUBLIC ${VTM_EXTERNAL_DIR}/vulkan-headers/include)
target_compile_definitions(VulkanCppModule PUBLIC VULKAN_HPP_NO_SMART_HANDLE)
target_compile_definitions(VulkanCppModule PUBLIC VK_NO_PROTOTYPES)
if(WIN32)
    target_compile_definitions(VulkanCppModule PUBLIC NOMINMAX)
    target_compile_definitions(VulkanCppModule PUBLIC VK_USE_PLATFORM_WIN32_KHR)
    target_compile_definitions(VulkanCppModule PUBLIC GLFW_EXPOSE_NATIVE_WIN32)
elseif(APPLE)
elseif(UNIX)
endif()

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory("${VTM_EXTERNAL_DIR}/glfw" EXCLUDE_FROM_ALL)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(VulkanCppModule PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/vulkan")
set_target_properties(glfw update_mappings PROPERTIES FOLDER "${VTM_EXTERNAL_FOLDER}/glfw")

#================================#
# Main Target                    #
#================================#
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE VulkanCppModule glfw)
add_dependencies(${PROJECT_NAME} VulkanCppModule glfw)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

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