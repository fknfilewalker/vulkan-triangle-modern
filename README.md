## The meme is dead

Render a vulkan triangle in ~380 lines of [code](https://github.com/fknfilewalker/vulkan-triangle-modern/blob/sdl/src/main.cpp)! (~370 without the shader code) 

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
* [SDL3](https://github.com/libsdl-org/SDL.git) for window handling

### How to build (on windows)
Use CMake for project configuration. The included `make.bat` script can be used for this. The Vulkan SDK is not required to run this code. Only for validation layers a Vulkan SDK installation is necessary.

> Please clone this repository with submodule!

### Notes
* Only works with NVIDIA and RADV driver, as others still need to implement `VK_EXT_swapchain_maintenance1`
* Linux should work (LLVM version >= 18.0.0 + Ninja build version >= 1.11)
* On MacOS use a newer LLVM/Clang version from brew
