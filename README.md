# The meme is dead

Render a vulkan triangle in ~390 lines of code! (~350 without the shader code) 

## Features
* Vulkan hpp headers using c++ modules
* Dynamic rendering
* Shader objects
* Bindless rendering using buffer references
* Resizable BAR for device local buffer access (needs to be activated in the bios)
* Modular code

## How to build (on windows)
Use CMake for project configuration. The included `make.bat` script can be used for this. The Vulkan SDK is not required to run this code. Only for validation layers a Vulkan SDK installation is necessary.

> Please clone this repository with submodule!

### Notes
* Could not get GLFW to handle all of window creation because of GLFW / Vulkan c++ module interoperability problems. It is also possible that I am doing something wrong.
* Linux window handling is not yet implemented because of the reason stated above.
* MacOS does not support c++ modules yet.
* All of the mentioned problems go away if the module import `import vulkan_hpp;` is replaced with `#include <vulkan/vulkan_raii.hpp>`