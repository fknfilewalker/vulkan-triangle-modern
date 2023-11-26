import vulkan_hpp;
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <bitset>
#include <fstream>
#include <vector>
#include <iostream>
#include <unordered_map>

constexpr bool fullscreen = false;
constexpr int window_width = 800;
constexpr int window_height = 600;

void exitWithError(const std::string_view error) {
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
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryRequirements.memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
	}
	return std::nullopt;
}

std::optional<uint32_t> findQueueFamilyIndex(const std::vector<vk::QueueFamilyProperties>& queueFamiliesProperties, vk::QueueFlags queueFlags) {
    std::optional<uint32_t> bestFamily;
    std::bitset<12> bestScore = 0;
    for (uint32_t i = 0; i < queueFamiliesProperties.size(); i++) {
        const std::bitset<12> score = static_cast<uint32_t>(queueFamiliesProperties[i].queueFlags);
		// check if queue family supports all requested queue flags
        if (static_cast<uint32_t>(queueFamiliesProperties[i].queueFlags & queueFlags) == static_cast<uint32_t>(queueFlags)) {
            // use queue family with the least other bits set
            if (!bestFamily.has_value() || score.count() < bestScore.count()) {
                bestFamily = i;
                bestScore = score;
            }
        }
    }
    return bestFamily;
}

struct Device
{
    Device(const vk::raii::PhysicalDevice& physicalDevice, const std::vector<const char*>& extensions, 
        const std::unordered_map<uint16_t, uint16_t>& queues, const void* pNext) : device{nullptr}
    {
	    constexpr float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
        deviceQueueCreateInfos.reserve(queues.size());
        for (const auto& [queueFamilyIndex, queueCount] : queues)
        {
        	deviceQueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queueCount, &priority });
		}
        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.setQueueCreateInfos(deviceQueueCreateInfos);
        deviceCreateInfo.setPEnabledExtensionNames(extensions);
        deviceCreateInfo.setPNext(pNext);
    	device = std::move(vk::raii::Device{ physicalDevice, deviceCreateInfo });

        for (const auto& [queueFamilyIndex, queueCount] : queues) {
        	std::vector<vk::raii::Queue> queueFamily;
			queueFamily.reserve(queueCount);
			for (uint16_t i = 0; i < queueCount; ++i) queueFamily.emplace_back(device.getQueue(queueFamilyIndex, i));
			queue.emplace_back(std::move(queueFamily));
		}
	}
	operator const vk::raii::Device&() const { return device; }

	vk::raii::Device device;
    std::vector<std::vector<vk::raii::Queue>> queue;
    //vk::raii::PhysicalDevice physicalDevice;
};

struct Buffer
{
    Buffer(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& memoryProperties,
        const vk::DeviceSize size, vk::BufferUsageFlags usageFlags, const vk::MemoryPropertyFlags memoryPropertiesFlags)
		: buffer{nullptr}, memory{nullptr}, deviceAddress{0}
    {
        usageFlags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const vk::BufferCreateInfo bufferCreateInfo{ {}, size, usageFlags };
        buffer = std::move(vk::raii::Buffer{ device, bufferCreateInfo });

        const auto memoryRequirements = buffer.getMemoryRequirements();
        const auto memoryTypeIndex = findMemoryTypeIndex(memoryProperties, memoryRequirements, memoryPropertiesFlags);
        if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");
        constexpr vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        const vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value(), &memoryAllocateFlagsInfo };
        memory = std::move(vk::raii::DeviceMemory{ device, memoryAllocateInfo });
        buffer.bindMemory(*memory, 0);

        const vk::BufferDeviceAddressInfo bufferDeviceAddressInfo { *buffer };
        deviceAddress = device.getBufferAddress(bufferDeviceAddressInfo);
	}
	vk::raii::Buffer buffer;
	vk::raii::DeviceMemory memory;
	vk::DeviceAddress deviceAddress;
};

struct Swapchain
{
	Swapchain(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface, uint32_t queueFamilyIndex) : swapchainKHR{nullptr},
		commandPool{ device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex } },
		commandBuffers{nullptr}
	{
		const auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		const auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto surfacePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        const uint32_t imageCount = std::min(3u, surfaceCapabilities.maxImageCount);
		const vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR {{}, *surface, imageCount,
			surfaceFormats[0].format, surfaceFormats[0].colorSpace, surfaceCapabilities.currentExtent,
			1u, vk::ImageUsageFlagBits::eColorAttachment};
		swapchainKHR = std::move(vk::raii::SwapchainKHR{ device, swapchainCreateInfoKHR });

        commandBuffers = std::move(vk::raii::CommandBuffers{ device, { *commandPool, vk::CommandBufferLevel::ePrimary, imageCount } });
        frames.reserve(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) frames.emplace_back(device, (commandBuffers[i]));
    }
    struct Frame {
        Frame(const vk::raii::Device& device, vk::raii::CommandBuffer& commandBuffer) : inFlightFence{ nullptr }, nextImageAvailableSemaphore{ nullptr },
			renderFinishedSemaphore{ nullptr }, commandBuffer{ commandBuffer }
        {
            inFlightFence = std::move(device.createFence(vk::FenceCreateInfo{ vk::FenceCreateFlagBits::eSignaled }));
            nextImageAvailableSemaphore = std::move(device.createSemaphore(vk::SemaphoreCreateInfo{}));
            renderFinishedSemaphore = std::move(device.createSemaphore(vk::SemaphoreCreateInfo{}));
        }
        vk::raii::Fence inFlightFence;
        vk::raii::Semaphore nextImageAvailableSemaphore;
        vk::raii::Semaphore renderFinishedSemaphore;
        vk::raii::CommandBuffer& commandBuffer;
    };

    Frame& getCurrentFrame() { return frames[currentFrame]; }
    void acquireNextImage(const vk::raii::Device& device) {
	    const auto nextImage = swapchainKHR.acquireNextImage(0, *frames[currentImageIdx].nextImageAvailableSemaphore);
        resultCheck(nextImage.first, "acquireNextImage error");
    	previousImageIdx = currentImageIdx;
        currentImageIdx = nextImage.second;

	    const Frame& frame = frames[currentImageIdx];
        while (vk::Result::eTimeout == device.waitForFences({ *frame.inFlightFence }, vk::True, UINT64_MAX)) {}
        device.resetFences({ *frame.inFlightFence });
        frame.commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }
    void submitImage(const vk::raii::Device& device)
    {
        const Frame& frame = frames[currentImageIdx];
        frame.commandBuffer.end();

        vk::raii::Queue queue = device.getQueue(0, 0);

        std::vector<vk::PipelineStageFlags> waitDstStageMask = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphores({ *frames[previousImageIdx].nextImageAvailableSemaphore });
        submitInfo.setWaitDstStageMask(waitDstStageMask);
        submitInfo.setSignalSemaphores({ *frame.renderFinishedSemaphore });
        submitInfo.setCommandBuffers({ *frame.commandBuffer });
        queue.submit({ submitInfo }, *frame.inFlightFence);
        
        vk::PresentInfoKHR presentInfoKHR;
        presentInfoKHR.setSwapchainCount(1);
        presentInfoKHR.setSwapchains({ *swapchainKHR });
        presentInfoKHR.setWaitSemaphores({ *frame.renderFinishedSemaphore });
        presentInfoKHR.setPImageIndices(&currentImageIdx);

        resultCheck(queue.presentKHR(presentInfoKHR), "present Swapchain error");
        
	}

    uint32_t currentFrame = 0;
    uint32_t currentImageIdx = 0;
    uint32_t previousImageIdx = 0;
    std::vector<Frame> frames;
    vk::raii::SwapchainKHR swapchainKHR;
    vk::raii::CommandPool commandPool;
    vk::raii::CommandBuffers commandBuffers;
};

std::string loadFile(const std::string_view path)
{
    std::ifstream in { path.data() };
    return { (std::istreambuf_iterator{in}),std::istreambuf_iterator<char>{} };
}

const char* vertexShader = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

layout(push_constant, std430) uniform pushConstant
{
    uint64_t vertexPtr;
};
layout(buffer_reference, std430) readonly buffer Vertex
{
    vec4 position;
};

void main() {
	Vertex vertex = Vertex(vertexPtr + gl_VertexIndex);
	gl_Position = vec4(vertex.position.xy, 0.0, 1.0);
})";

const char* fragmentShader = R"(
#version 450
void main() {
	gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
})";

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
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) iLayers.clear();
#endif
    if (!extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions)) exitWithError("Instance extensions not available");
    
    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(iExtensions);
    instanceCreateInfo.setPEnabledLayerNames(iLayers);
#ifdef __APPLE__
    instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
#endif
    vk::raii::Instance instance(context, instanceCreateInfo);

	// unfortunately glfw surface creation does not work with the vulkan c++20 modules
    vk::raii::SurfaceKHR surfaceKHR{ nullptr };
#ifdef _WIN32
	vk::Win32SurfaceCreateInfoKHR win32SurfaceCreateInfoKHR{ {}, nullptr, glfwGetWin32Window(window) };
    surfaceKHR = std::move(vk::raii::SurfaceKHR { instance, win32SurfaceCreateInfoKHR });
#endif

    vk::raii::PhysicalDevices physicalDevices{ instance };
    const vk::raii::PhysicalDevice physicalDevice{ std::move(physicalDevices[0]) };

    // queue
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    const auto queueFamilyIndex = findQueueFamilyIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics);
    if (!queueFamilyIndex.has_value()) exitWithError("No queue family index found");
    // extensions
    std::vector dExtensions { vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName };
    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
	// features
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ true };
    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ true, &bufferDeviceAddressFeatures };
    Device device{ physicalDevice, dExtensions, {{queueFamilyIndex.value(), 1}}, &shaderObjectFeatures };

	// setup vertex buffer
    std::vector vertices = {
    	-0.5f, -0.5f, 0.0f, 1.0f,
    	 0.5f, -0.5f, 0.0f, 1.0f,
    	 0.0f,  0.5f, 0.0f, 1.0f
    };
    const size_t verticesSize = vertices.size() * sizeof(float);
    auto memoryProperties = physicalDevice.getMemoryProperties();
    Buffer buffer{ device, memoryProperties, verticesSize, vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible};
    void* p = buffer.memory.mapMemory(0, vk::WholeSize);
    std::memcpy(p, vertices.data(), verticesSize);
    buffer.memory.unmapMemory();

    // setup shader objects
	// https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_object.adoc
    vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint64_t) };

    vk::ShaderCreateInfoEXT shaderCreateInfoVertex = { vk::ShaderCreateFlagBitsEXT::eLinkStage, vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
    shaderCreateInfoVertex.codeType = vk::ShaderCodeTypeEXT::eSpirv;
    shaderCreateInfoVertex.pName = "main";
    shaderCreateInfoVertex.pCode = reinterpret_cast<const uint32_t*>(vertexShader);
    shaderCreateInfoVertex.codeSize = std::strlen(vertexShader);
    shaderCreateInfoVertex.setPushConstantRanges({ pcRange });

    vk::ShaderCreateInfoEXT shaderCreateInfoFragment = { vk::ShaderCreateFlagBitsEXT::eLinkStage, vk::ShaderStageFlagBits::eFragment};
    shaderCreateInfoVertex.codeType = vk::ShaderCodeTypeEXT::eSpirv;
    shaderCreateInfoVertex.pName = "main";
    shaderCreateInfoVertex.pCode = reinterpret_cast<const uint32_t*>(fragmentShader);
    shaderCreateInfoVertex.codeSize = std::strlen(fragmentShader);

    const vk::ShaderCreateInfoEXT sci[2] = { shaderCreateInfoVertex, shaderCreateInfoFragment };
    //vk::raii::ShaderEXT shader{ device, sci[0] };

    Swapchain swapchain{ device, physicalDevice, surfaceKHR, queueFamilyIndex.value() };

    while(!glfwWindowShouldClose(window))
    {
    	glfwPollEvents();
	}
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
