import vulkan_hpp;
#include <GLFW/glfw3.h>

#include <algorithm>
#include <fstream>
#include <vector>
#include <iostream>

constexpr bool fullscreen = false;
constexpr int window_width = 800;
constexpr int window_height = 600;

vk::Bool32 debugMessageFunc(vk::DebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
						    vk::DebugUtilsMessageTypeFlagsEXT /*messageTypes*/,
						    vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData, 
						    void* /*pUserData*/) {
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
            if constexpr (std::is_same_v<vk::LayerProperties, T>) return std::string_view{ availableElement.layerName.data() }.compare(requestedElement) == 0;
            else if constexpr (std::is_same_v<vk::ExtensionProperties, T>) return std::string_view{ availableElement.extensionName.data() }.compare(requestedElement) == 0;
            else return false;
        }) != available.end();
    });
}

std::optional<uint32_t> findMemoryTypeIndex(const vk::PhysicalDeviceMemoryProperties& memoryProperties, 
	const vk::MemoryRequirements& memoryRequirements, const vk::MemoryPropertyFlags properties) {
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
	{
		if ((memoryRequirements.memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
	}
	return std::nullopt;
}

struct Buffer
{
    Buffer(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& memoryProperties,
		vk::BufferUsageFlags usageFlags, const vk::DeviceSize size, const vk::MemoryPropertyFlags memoryPropertiesFlags)
		: buffer{nullptr}, deviceMemory{nullptr}, deviceAddress{0}
    {
        usageFlags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const vk::BufferCreateInfo bufferCreateInfo{ {}, size, usageFlags };
        buffer = std::move(vk::raii::Buffer{ device, bufferCreateInfo });
        const auto memoryRequirements = buffer.getMemoryRequirements();
        const auto memoryTypeIndex = findMemoryTypeIndex(memoryProperties, memoryRequirements, memoryPropertiesFlags);
        if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");

        constexpr vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        const vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value(), &memoryAllocateFlagsInfo };
    	deviceMemory = std::move(vk::raii::DeviceMemory{ device, memoryAllocateInfo });
        buffer.bindMemory(*deviceMemory, 0);

        const vk::BufferDeviceAddressInfo bufferDeviceAddressInfo { *buffer };
        deviceAddress = device.getBufferAddress(bufferDeviceAddressInfo);
	}   

	vk::raii::Buffer buffer;
	vk::raii::DeviceMemory deviceMemory;
	vk::DeviceAddress deviceAddress;
};

std::string loadFile(const std::string_view path)
{
    std::ifstream in { path.data() };
    return { (std::istreambuf_iterator{in}),std::istreambuf_iterator<char>{} };
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
    if (!extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions)) exitWithError("Instance extensions not available");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) exitWithError("Instance layers not available");

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

    vk::raii::PhysicalDevices physicalDevices{ instance };
    const vk::raii::PhysicalDevice physicalDevice{ std::move(physicalDevices[0]) };
    auto memoryProperties = physicalDevice.getMemoryProperties();
    // queue
    constexpr float priority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ {}, 0, 1, &priority };
    // extensions
    std::vector dExtensions { vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName };
    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
	// features
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ true };

    vk::DeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.setQueueCreateInfos({ deviceQueueCreateInfo });
    deviceCreateInfo.setPEnabledExtensionNames(dExtensions);
    deviceCreateInfo.setPNext(&bufferDeviceAddressFeatures);
    vk::raii::Device device{ physicalDevice, deviceCreateInfo };

    /*vk::MemoryPropertyFlags memoryPropertiesFlags = vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible;
    vk::BufferCreateInfo bufferCreateInfo{ {}, 10, vk::BufferUsageFlagBits::eVertexBuffer };
    vk::raii::Buffer buffer{ device, bufferCreateInfo };
    auto memoryRequirements = buffer.getMemoryRequirements();
    auto memoryTypeIndex = findMemoryTypeIndex(memoryProperties, memoryRequirements, memoryPropertiesFlags);
    if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");
    vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value() };
    vk::raii::DeviceMemory deviceMemory{ device, memoryAllocateInfo };*/

    Buffer buffer{ device, memoryProperties, vk::BufferUsageFlagBits::eVertexBuffer, 10, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible };

    vk::ShaderCreateInfoEXT shaderCreateInfoVertex = { vk::ShaderCreateFlagBitsEXT::eLinkStage, vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
    //vk::raii::ShaderEXT vertexShader{ device, shaderCreateInfo };

    while(!glfwWindowShouldClose(window))
    {
    	glfwPollEvents();
	}
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}