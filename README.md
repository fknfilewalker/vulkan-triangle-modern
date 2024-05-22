## The meme is dead

Render a vulkan triangle in ~390 lines of [code](https://github.com/fknfilewalker/vulkan-triangle-modern/blob/swapchain-maintenance/src/main.cpp)! (~350 without the shader code) 

### Highlights
* Vulkan hpp headers included as a c++ module (c++20 required)
* Dynamic rendering (`VK_KHR_dynamic_rendering`)
* Shader objects (`VK_EXT_shader_object`)
* Bindless rendering using buffer references (`VK_EXT_buffer_device_address`)
* (Resizable) BAR for device local buffer access
* Deferred swapchain image allocation (`VK_EXT_swapchain_maintenance1`)
* Straightforward swapchain sync (`VK_EXT_swapchain_maintenance1`)
* Modular code

### How to build (on windows)
Use CMake for project configuration. The included `make.bat` script can be used for this. The Vulkan SDK is not required to run this code. Only for validation layers a Vulkan SDK installation is necessary.

> Please clone this repository with submodule!

### Notes
* Could not get GLFW to handle all of window creation because of GLFW / Vulkan c++ module interoperability problems. It is also possible that I am doing something wrong.
* Linux window handling is not yet implemented because of the reason stated above.
* MacOS does not support c++ modules yet.
* All of the mentioned problems go away if the module import `import vulkan_hpp;` is replaced with `#include <vulkan/vulkan_raii.hpp>`
