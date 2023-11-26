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
		// check if queue family supports all requested queue flags
        if (static_cast<uint32_t>(queueFamiliesProperties[i].queueFlags & queueFlags) == static_cast<uint32_t>(queueFlags)) {
			const std::bitset<12> score = static_cast<uint32_t>(queueFamiliesProperties[i].queueFlags);
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
           const std::unordered_map<uint16_t, uint16_t>& queues, const void* pNext) : device{nullptr}, physicalDevice{physicalDevice}
    {
	    constexpr float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
        deviceQueueCreateInfos.reserve(queues.size());
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
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
    operator const vk::raii::PhysicalDevice&() const { return physicalDevice; }

	vk::raii::Device device;
    std::vector<std::vector<vk::raii::Queue>> queue;
    vk::raii::PhysicalDevice physicalDevice;
};

struct Buffer
{
    Buffer(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& memoryProperties, const vk::DeviceSize size,
           const vk::BufferUsageFlags usageFlags, const vk::MemoryPropertyFlags memoryPropertiesFlags)
		: buffer{ device, { {}, size, usageFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress } }, memory{nullptr}
    {
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
    // Data for one frame/image in our swapchain
    struct Frame {
        Frame(const vk::raii::Device& device, const vk::Image& image, vk::raii::CommandBuffer& commandBuffer) :
    		image{ image }, imageView{nullptr}, inFlightFence{ device, vk::FenceCreateInfo{ vk::FenceCreateFlagBits::eSignaled } },
    		nextImageAvailableSemaphore{ device, vk::SemaphoreCreateInfo{} }, renderFinishedSemaphore{ device, vk::SemaphoreCreateInfo{} }, commandBuffer{ commandBuffer }
        {
            imageView = std::move(vk::raii::ImageView{ device, vk::ImageViewCreateInfo{ {}, image, vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Unorm, 
                {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } } });
        }
        vk::Image image;
        vk::raii::ImageView imageView;
        vk::raii::Fence inFlightFence;
        vk::raii::Semaphore nextImageAvailableSemaphore;
        vk::raii::Semaphore renderFinishedSemaphore;
        vk::raii::CommandBuffer& commandBuffer;
    };

	Swapchain(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface, const uint32_t queueFamilyIndex) : currentImageIdx{0}, previousImageIdx{0}, swapchainKHR{nullptr},
		commandPool{ device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex } }, commandBuffers{nullptr}
	{
		const auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		const auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto surfacePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        imageCount = std::min(3u, surfaceCapabilities.maxImageCount);
		const vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR {{}, *surface, imageCount,
			surfaceFormats[0].format, surfaceFormats[0].colorSpace, surfaceCapabilities.currentExtent,
			1u, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst};
		swapchainKHR = std::move(vk::raii::SwapchainKHR{ device, swapchainCreateInfoKHR });

        commandBuffers = std::move(vk::raii::CommandBuffers{ device, { *commandPool, vk::CommandBufferLevel::ePrimary, imageCount } });
        std::vector<vk::Image> images = swapchainKHR.getImages();
        frames.reserve(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) frames.emplace_back(device, images[i], commandBuffers[i]);
    }

    const Frame& getCurrentFrame() { return frames[currentImageIdx]; }
    const Frame& getPreviousFrame() { return frames[previousImageIdx]; }
    void acquireNextImage(const vk::raii::Device& device) {
	    const auto nextImage = swapchainKHR.acquireNextImage(0, *frames[currentImageIdx].nextImageAvailableSemaphore);
        resultCheck(nextImage.first, "acquireing next swapchain image error");
    	previousImageIdx = currentImageIdx;
        currentImageIdx = nextImage.second;

	    const Frame& frame = frames[currentImageIdx];
        while (vk::Result::eTimeout == device.waitForFences({ *frame.inFlightFence }, vk::True, UINT64_MAX)) {}
        device.resetFences({ *frame.inFlightFence });
        frame.commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }
    void submitImage(const vk::raii::Queue& presentQueue)
    {
        const Frame& curFrame = getCurrentFrame();
        curFrame.commandBuffer.end();

        constexpr vk::PipelineStageFlags waitDstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo {};
        submitInfo.setWaitSemaphores({ *getPreviousFrame().nextImageAvailableSemaphore });
        submitInfo.setPWaitDstStageMask(&waitDstStageMask);
        submitInfo.setSignalSemaphores({ *curFrame.renderFinishedSemaphore });
        submitInfo.setCommandBuffers({ *curFrame.commandBuffer });
        presentQueue.submit({ submitInfo }, *curFrame.inFlightFence);
        
        vk::PresentInfoKHR presentInfoKHR { { *curFrame.renderFinishedSemaphore }};
        presentInfoKHR.setSwapchains({ *swapchainKHR });
        presentInfoKHR.setPImageIndices(&currentImageIdx);

        resultCheck(presentQueue.presentKHR(presentInfoKHR), "present swapchain image error");
	}

    uint32_t imageCount;
    uint32_t currentImageIdx;
    uint32_t previousImageIdx;
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
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

const uint64_t sizeOfFloat = 4ul;
const uint64_t sizeOfVec3 = 3ul * sizeOfFloat;

layout(buffer_reference, scalar) readonly buffer Vertex
{
    vec3 position;
};
layout(push_constant, scalar) uniform pushConstant
{
    uint64_t vertexPtr;
};

void main() {
	Vertex vertex = Vertex(vertexPtr + sizeOfVec3 * gl_VertexIndex);
	gl_Position = vec4(vertex.position.xy, 0.0, 1.0);
})";

const char* fragmentShader = R"(
#version 450
layout (location = 0) out vec4 fragColor;
void main() {
	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
})";


// glslangValidator -V -o vertexShader.h --vn vertShader triangle.vert
const uint32_t vertexShaderSPV[] = {
    0x07230203,0x00010000,0x0008000b,0x00000034,0x00000000,0x00020011,0x00000001,0x00020011,
    0x0000000b,0x00020011,0x000014e3,0x0009000a,0x5f565053,0x5f52484b,0x73796870,0x6c616369,
    0x6f74735f,0x65676172,0x6675625f,0x00726566,0x0006000b,0x00000001,0x4c534c47,0x6474732e,
    0x3035342e,0x00000000,0x0003000e,0x000014e4,0x00000001,0x0007000f,0x00000000,0x00000004,
    0x6e69616d,0x00000000,0x00000017,0x00000025,0x00030003,0x00000002,0x000001c2,0x00070004,
    0x455f4c47,0x625f5458,0x65666675,0x65725f72,0x65726566,0x0065636e,0x00080004,0x455f4c47,
    0x625f5458,0x65666675,0x65725f72,0x65726566,0x3265636e,0x00000000,0x00080004,0x455f4c47,
    0x735f5458,0x616c6163,0x6c625f72,0x5f6b636f,0x6f79616c,0x00007475,0x000d0004,0x455f4c47,
    0x735f5458,0x65646168,0x78655f72,0x63696c70,0x615f7469,0x68746972,0x6974656d,0x79745f63,
    0x5f736570,0x36746e69,0x00000034,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00040005,
    0x00000009,0x74726556,0x00007865,0x00060006,0x00000009,0x00000000,0x69736f70,0x6e6f6974,
    0x00000000,0x00040005,0x0000000b,0x74726576,0x00007865,0x00060005,0x0000000d,0x68737570,
    0x736e6f43,0x746e6174,0x00000000,0x00060006,0x0000000d,0x00000000,0x74726576,0x74507865,
    0x00000072,0x00030005,0x0000000f,0x00000000,0x00060005,0x00000017,0x565f6c67,0x65747265,
    0x646e4978,0x00007865,0x00060005,0x00000023,0x505f6c67,0x65567265,0x78657472,0x00000000,
    0x00060006,0x00000023,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x00000023,
    0x00000001,0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x00000023,0x00000002,
    0x435f6c67,0x4470696c,0x61747369,0x0065636e,0x00070006,0x00000023,0x00000003,0x435f6c67,
    0x446c6c75,0x61747369,0x0065636e,0x00030005,0x00000025,0x00000000,0x00040048,0x00000009,
    0x00000000,0x00000018,0x00050048,0x00000009,0x00000000,0x00000023,0x00000000,0x00030047,
    0x00000009,0x00000002,0x00030047,0x0000000b,0x000014ec,0x00050048,0x0000000d,0x00000000,
    0x00000023,0x00000000,0x00030047,0x0000000d,0x00000002,0x00040047,0x00000017,0x0000000b,
    0x0000002a,0x00050048,0x00000023,0x00000000,0x0000000b,0x00000000,0x00050048,0x00000023,
    0x00000001,0x0000000b,0x00000001,0x00050048,0x00000023,0x00000002,0x0000000b,0x00000003,
    0x00050048,0x00000023,0x00000003,0x0000000b,0x00000004,0x00030047,0x00000023,0x00000002,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030027,0x00000006,0x000014e5,
    0x00030016,0x00000007,0x00000020,0x00040017,0x00000008,0x00000007,0x00000003,0x0003001e,
    0x00000009,0x00000008,0x00040020,0x00000006,0x000014e5,0x00000009,0x00040020,0x0000000a,
    0x00000007,0x00000006,0x00040015,0x0000000c,0x00000040,0x00000000,0x0003001e,0x0000000d,
    0x0000000c,0x00040020,0x0000000e,0x00000009,0x0000000d,0x0004003b,0x0000000e,0x0000000f,
    0x00000009,0x00040015,0x00000010,0x00000020,0x00000001,0x0004002b,0x00000010,0x00000011,
    0x00000000,0x00040020,0x00000012,0x00000009,0x0000000c,0x0005002b,0x0000000c,0x00000015,
    0x0000000c,0x00000000,0x00040020,0x00000016,0x00000001,0x00000010,0x0004003b,0x00000016,
    0x00000017,0x00000001,0x00040015,0x00000019,0x00000040,0x00000001,0x00040017,0x0000001f,
    0x00000007,0x00000004,0x00040015,0x00000020,0x00000020,0x00000000,0x0004002b,0x00000020,
    0x00000021,0x00000001,0x0004001c,0x00000022,0x00000007,0x00000021,0x0006001e,0x00000023,
    0x0000001f,0x00000007,0x00000022,0x00000022,0x00040020,0x00000024,0x00000003,0x00000023,
    0x0004003b,0x00000024,0x00000025,0x00000003,0x00040017,0x00000027,0x00000007,0x00000002,
    0x00040020,0x00000028,0x000014e5,0x00000008,0x0004002b,0x00000007,0x0000002c,0x00000000,
    0x0004002b,0x00000007,0x0000002d,0x3f800000,0x00040020,0x00000031,0x00000003,0x0000001f,
    0x0005002b,0x0000000c,0x00000033,0x00000004,0x00000000,0x00050036,0x00000002,0x00000004,
    0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,0x0000000a,0x0000000b,0x00000007,
    0x00050041,0x00000012,0x00000013,0x0000000f,0x00000011,0x0004003d,0x0000000c,0x00000014,
    0x00000013,0x0004003d,0x00000010,0x00000018,0x00000017,0x00040072,0x00000019,0x0000001a,
    0x00000018,0x0004007c,0x0000000c,0x0000001b,0x0000001a,0x00050084,0x0000000c,0x0000001c,
    0x00000015,0x0000001b,0x00050080,0x0000000c,0x0000001d,0x00000014,0x0000001c,0x00040078,
    0x00000006,0x0000001e,0x0000001d,0x0003003e,0x0000000b,0x0000001e,0x0004003d,0x00000006,
    0x00000026,0x0000000b,0x00050041,0x00000028,0x00000029,0x00000026,0x00000011,0x0006003d,
    0x00000008,0x0000002a,0x00000029,0x00000002,0x00000004,0x0007004f,0x00000027,0x0000002b,
    0x0000002a,0x0000002a,0x00000000,0x00000001,0x00050051,0x00000007,0x0000002e,0x0000002b,
    0x00000000,0x00050051,0x00000007,0x0000002f,0x0000002b,0x00000001,0x00070050,0x0000001f,
    0x00000030,0x0000002e,0x0000002f,0x0000002c,0x0000002d,0x00050041,0x00000031,0x00000032,
    0x00000025,0x00000011,0x0003003e,0x00000032,0x00000030,0x000100fd,0x00010038
};

const uint32_t fragmentShaderSPV[] = {
    0x07230203,0x00010000,0x0008000b,0x0000000d,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00050005,0x00000009,0x67617266,0x6f6c6f43,0x00000072,0x00040047,0x00000009,0x0000001e,
    0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,
    0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,
    0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,0x0004002b,0x00000006,0x0000000a,
    0x3f800000,0x0004002b,0x00000006,0x0000000b,0x00000000,0x0007002c,0x00000007,0x0000000c,
    0x0000000a,0x0000000b,0x0000000b,0x0000000a,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0003003e,0x00000009,0x0000000c,0x000100fd,0x00010038
};

int main(int /*argc*/, char* /*argv[]*/)
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
    uint32_t glfwInstanceExtensionCount;
    const char** glfwInstanceExtensionNames = glfwGetRequiredInstanceExtensions(&glfwInstanceExtensionCount);
    iExtensions.reserve(glfwInstanceExtensionCount + 1);
    for (uint32_t i = 0; i < glfwInstanceExtensionCount; ++i) iExtensions.emplace_back(glfwInstanceExtensionNames[i]);

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

    // Device setup
    // * find queue
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    const auto queueFamilyIndex = findQueueFamilyIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics);
    if (!queueFamilyIndex.has_value()) exitWithError("No queue family index found");
    if (!physicalDevice.getSurfaceSupportKHR(queueFamilyIndex.value(), *surfaceKHR)) exitWithError("Queue family does not support presentation");
    // * check extensions
    std::vector dExtensions { vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName };
    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
	// * activate features
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ true };
    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ true, &bufferDeviceAddressFeatures };
    vk::PhysicalDeviceSynchronization2Features synchronization2Features{ true, &shaderObjectFeatures };
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{ true, &synchronization2Features };
    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2{ {}, &dynamicRenderingFeatures };
    physicalDeviceFeatures2.features.shaderInt64 = true;
    // * create device
    Device device{ physicalDevice, dExtensions, {{queueFamilyIndex.value(), 1}}, &physicalDeviceFeatures2 };

	// Vertex buffer setup
    std::vector vertices = {
    	-0.5f, -0.5f, 0.0f,
    	 0.5f, -0.5f, 0.0f,
    	 0.0f,  0.5f, 0.0f
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
    shaderCreateInfoVertex.pCode = vertexShaderSPV;
    shaderCreateInfoVertex.codeSize = sizeof(vertexShaderSPV);
    shaderCreateInfoVertex.setPushConstantRanges({ pcRange });

    vk::ShaderCreateInfoEXT shaderCreateInfoFragment = { vk::ShaderCreateFlagBitsEXT::eLinkStage, vk::ShaderStageFlagBits::eFragment, {}};
    shaderCreateInfoFragment.codeType = vk::ShaderCodeTypeEXT::eSpirv;
    shaderCreateInfoFragment.pName = "main";
    shaderCreateInfoFragment.pCode = fragmentShaderSPV;
    shaderCreateInfoFragment.codeSize = sizeof(fragmentShaderSPV);
    shaderCreateInfoFragment.setPushConstantRanges({ pcRange });
    std::vector<vk::raii::ShaderEXT> shaders = device.device.createShadersEXT({ shaderCreateInfoVertex, shaderCreateInfoFragment });
    vk::raii::PipelineLayout layout{ device, {{}, 0, {}, 1, &pcRange} };

    Swapchain swapchain{ device, physicalDevice, surfaceKHR, queueFamilyIndex.value() };
    constexpr vk::ImageSubresourceRange imageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    vk::ImageMemoryBarrier2 imageMemoryBarrier{};
    imageMemoryBarrier.subresourceRange = imageSubresourceRange;
    vk::DependencyInfoKHR dependencyInfo{};
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageMemoryBarrier;

    while(!glfwWindowShouldClose(window))
    {
    	glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        swapchain.acquireNextImage(device);
        const auto& cFrame = swapchain.getCurrentFrame();
        const auto& cmdBuffer = cFrame.commandBuffer;

        imageMemoryBarrier.image = cFrame.image;
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        cmdBuffer.pipelineBarrier2(dependencyInfo);

        vk::RenderingAttachmentInfo rAttachmentInfo { *cFrame.imageView, vk::ImageLayout::eColorAttachmentOptimal };
        rAttachmentInfo.clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
        rAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        rAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;

        cmdBuffer.beginRendering({ {}, { {}, {window_width, window_height} }, 1, 0, 1, &rAttachmentInfo });
        {
            cmdBuffer.bindShadersEXT({ vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment }, { *shaders[0], *shaders[1] });
            cmdBuffer.pushConstants<uint64_t>(*layout, vk::ShaderStageFlagBits::eVertex, 0, { buffer.deviceAddress });
            cmdBuffer.setPrimitiveTopologyEXT(vk::PrimitiveTopology::eTriangleList);
            cmdBuffer.setPolygonModeEXT(vk::PolygonMode::eFill);
            cmdBuffer.setFrontFaceEXT(vk::FrontFace::eCounterClockwise);
            cmdBuffer.setCullModeEXT(vk::CullModeFlagBits::eNone);
            cmdBuffer.setColorWriteMaskEXT(0, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB);
            cmdBuffer.setSampleMaskEXT(vk::SampleCountFlagBits::e1, { 0xffffffff });
            cmdBuffer.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
            cmdBuffer.setViewportWithCountEXT({ { 0, 0, window_width, window_height } });
            cmdBuffer.setScissorWithCountEXT({ { {0, 0}, { window_width, window_height } } });
            cmdBuffer.setVertexInputEXT({}, {});
            cmdBuffer.setColorBlendEnableEXT(0, { vk::False });
            cmdBuffer.setDepthTestEnableEXT(vk::False);
            cmdBuffer.setDepthWriteEnableEXT(vk::False);
            cmdBuffer.setDepthBiasEnableEXT(vk::False);
            cmdBuffer.setStencilTestEnableEXT(vk::False);
            cmdBuffer.setRasterizerDiscardEnableEXT(vk::False);
            cmdBuffer.setAlphaToCoverageEnableEXT(vk::False);
            cmdBuffer.setPrimitiveRestartEnableEXT(vk::False);
            cmdBuffer.draw(3, 1, 0, 0);
        }
        cmdBuffer.endRendering();

        imageMemoryBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        imageMemoryBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        cmdBuffer.pipelineBarrier2(dependencyInfo);
        swapchain.submitImage(device.queue[queueFamilyIndex.value()][0]);
	}
    device.device.waitIdle();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
