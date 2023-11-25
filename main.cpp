import vulkan_hpp;
#include <GLFW/glfw3.h>

#include <algorithm>
#include <vector>
#include <iostream>

constexpr bool fullscreen = false;
constexpr int window_width = 800;
constexpr int window_height = 600;

class App {
    
};

vk::Bool32 debugMessageFunc(vk::DebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
						    vk::DebugUtilsMessageTypeFlagsEXT /*messageTypes*/,
						    vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData, 
						    void* /*pUserData*/)
{
    std::printf("validation layer: %s\n", pCallbackData->pMessage);
    return vk::False;
}

void exitWithError(const std::string_view error)
{
	std::printf("%s\n", error.data());
	exit(EXIT_FAILURE);
}

template<typename T>
bool extensionsOrLayersAvailable(const std::vector<T>& available, const std::vector<const char*>& requested) {
    return std::all_of(requested.begin(), requested.end(), [&available](const char* requestedElement) {
        return std::find_if(available.begin(), available.end(), [requestedElement](const T& availableElement){
            if constexpr (std::is_same_v<vk::LayerProperties, T>) return std::string_view{ availableElement.layerName.data() }.compare(requestedElement);
            else if constexpr (std::is_same_v<vk::ExtensionProperties, T>) return std::string_view{ availableElement.extensionName.data() }.compare(requestedElement);
            else return false;
        }) != available.end();
    });
}

int main(int argc, char *argv[])
{
    if (!glfwInit()) exitWithError("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWmonitor* monitor = nullptr;
    if (fullscreen) monitor = glfwGetPrimaryMonitor();
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Vulkan Triangle Modern", monitor, nullptr);

	const vk::raii::Context context;
    constexpr vk::ApplicationInfo applicationInfo{nullptr, 0, nullptr, 0, vk::ApiVersion13};
    
    std::vector<const char*> iExtensions;
    uint32_t glfw_instance_extensions_count;
    const char** glfw_instance_extensions_names = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);
    iExtensions.reserve(glfw_instance_extensions_count + 1);
    for (uint32_t i = 0; i < glfw_instance_extensions_count; ++i) iExtensions.emplace_back(glfw_instance_extensions_names[i]);

#ifdef __APPLE__
    iExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    std::vector<const char*> iLayers;
#if !defined( NDEBUG )
    iLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif
    bool extensionsAvailable = extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions);
    bool layersAvailable = extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers);

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(iExtensions);
    instanceCreateInfo.setPEnabledLayerNames(iLayers);
#ifdef __APPLE__
    instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
#endif
    vk::raii::Instance instance(context, instanceCreateInfo);

#if !defined( NDEBUG )
    const vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{ vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };
    const vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{ vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation };
    //const vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{ {}, severityFlags, messageTypeFlags, &debugMessageFunc };
    //vk::raii::DebugUtilsMessengerEXT debugUtilsMessengerEXT(instance, debugUtilsMessengerCreateInfo);
#endif

    vk::SurfaceKHR surface;
    //glfwCreateWindowSurface(instance, window, nullptr, &surface);

    auto devices = instance.enumeratePhysicalDevices();


    while(!glfwWindowShouldClose(window))
    {
    	glfwPollEvents();
	}
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
