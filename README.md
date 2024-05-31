## The meme is dead

Render a vulkan triangle in ~380 lines of [code](https://github.com/fknfilewalker/vulkan-triangle-modern/blob/main/src/main.cpp)! (~370 without the shader code) 

### Highlights
* Vulkan hpp headers included as a c++ module (c++20 required)
* Dynamic rendering (`VK_KHR_dynamic_rendering`)
* Shader objects (`VK_EXT_shader_object`)
* Bindless rendering using buffer references (`VK_EXT_buffer_device_address`)
* (Resizable) BAR for device local buffer access
* Deferred swapchain image allocation (`VK_EXT_swapchain_maintenance1`)
* Straightforward swapchain sync (`VK_EXT_swapchain_maintenance1`)
* [Slang](https://github.com/shader-slang/slang) used for shader code
* Modular code

### How to build (on windows)
Use CMake for project configuration. The included `make.bat` script can be used for this. The Vulkan SDK is not required to run this code. Only for validation layers a Vulkan SDK installation is necessary.

> Please clone this repository with submodule!

### Notes
* `VK_EXT_swapchain_maintenance1` still has some validation bugs (the semaphore validation error should go away with the next SDK update)
* Linux should work (LLVM Version >= 18.0.0 + Ninja build Version >= 1.11)
* Code will become shorter once I replace the macos fallback to just c++ modules
