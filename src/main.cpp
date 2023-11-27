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
constexpr uint32_t window_width = 800;
constexpr uint32_t window_height = 600;

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

#include "shaders.h"

void exitWithError(const std::string_view error) {
    std::printf("%s\n", error.data());
    exit(EXIT_FAILURE);
}

template<typename T>
bool extensionsOrLayersAvailable(const std::vector<T>& available, const std::vector<const char*>& requested) {
    static_assert(std::is_same_v<vk::LayerProperties, T> || std::is_same_v<vk::ExtensionProperties, T>);
    return std::all_of(requested.begin(), requested.end(), [&available](const char* requestedElement) {
        return std::find_if(available.begin(), available.end(), [requestedElement](const T& availableElement) {
            if constexpr (std::is_same_v<vk::LayerProperties, T>) return std::string_view{ availableElement.layerName.data() }.compare(requestedElement) == 0;
            else if constexpr (std::is_same_v<vk::ExtensionProperties, T>) return std::string_view{ availableElement.extensionName.data() }.compare(requestedElement) == 0;
            }) != available.end();
        });
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
        const std::unordered_map<uint16_t, uint16_t>& queues, const void* pNext) : device{ nullptr }, physicalDevice{ physicalDevice },
        memoryProperties{ physicalDevice.getMemoryProperties() }
    {
        constexpr float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
        deviceQueueCreateInfos.reserve(queues.size());
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
            deviceQueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queueCount, &priority });
        }
        const vk::DeviceCreateInfo deviceCreateInfo{ {}, deviceQueueCreateInfos, {}, extensions,{}, pNext };
        device = std::move(vk::raii::Device{ physicalDevice, deviceCreateInfo });
        // get queues
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
            std::vector<vk::raii::Queue> queueFamily;
            queueFamily.reserve(queueCount);
            for (uint16_t i = 0; i < queueCount; ++i) queueFamily.emplace_back(device.getQueue(queueFamilyIndex, i));
            queue.emplace_back(std::move(queueFamily));
        }
    }

    std::optional<uint32_t> findMemoryTypeIndex(const vk::MemoryRequirements& requirements, const vk::MemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
        return std::nullopt;
    }

    operator const vk::raii::Device& () const { return device; }
    operator const vk::raii::PhysicalDevice& () const { return physicalDevice; }

    vk::raii::Device device;
    std::vector<std::vector<vk::raii::Queue>> queue;
    const vk::raii::PhysicalDevice physicalDevice;
    const vk::PhysicalDeviceMemoryProperties memoryProperties;
};

struct Buffer
{
    Buffer(const Device& device, const vk::DeviceSize size, const vk::BufferUsageFlags usageFlags, const vk::MemoryPropertyFlags memoryPropertyFlags)
        : buffer{ device, { {}, size, usageFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress } }, memory{ nullptr }
    {
        const auto memoryRequirements = buffer.getMemoryRequirements();
        const auto memoryTypeIndex = device.findMemoryTypeIndex(memoryRequirements, memoryPropertyFlags);
        if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");
        constexpr vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        const vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value(), &memoryAllocateFlagsInfo };
        memory = std::move(vk::raii::DeviceMemory{ device, memoryAllocateInfo });
        buffer.bindMemory(*memory, 0);

        const vk::BufferDeviceAddressInfo bufferDeviceAddressInfo{ *buffer };
        deviceAddress = device.device.getBufferAddress(bufferDeviceAddressInfo);
    }
    const vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    vk::DeviceAddress deviceAddress;
};

struct Swapchain
{
    // Data for one frame/image in our swapchain
    struct Frame {
        Frame(const vk::raii::Device& device, const vk::Image& image, vk::raii::CommandBuffer& commandBuffer) :
            image{ image }, imageView{ nullptr }, inFlightFence{ device, vk::FenceCreateInfo{ vk::FenceCreateFlagBits::eSignaled } },
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

    Swapchain(const Device& device, const vk::raii::SurfaceKHR& surface, const uint32_t queueFamilyIndex) : currentImageIdx{ 0 }, previousImageIdx{ 0 }, swapchainKHR{ nullptr },
        commandPool{ device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex } }, commandBuffers{ nullptr }
    {
        const auto surfaceCapabilities = device.physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const auto surfaceFormats = device.physicalDevice.getSurfaceFormatsKHR(*surface);

        imageCount = std::min(3u, surfaceCapabilities.maxImageCount);
        extent = surfaceCapabilities.currentExtent;
        const vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR{ {}, *surface, imageCount,
            surfaceFormats[0].format, surfaceFormats[0].colorSpace, extent,
            1u, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst };
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
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores({ *getPreviousFrame().nextImageAvailableSemaphore });
        submitInfo.setPWaitDstStageMask(&waitDstStageMask);
        submitInfo.setSignalSemaphores({ *curFrame.renderFinishedSemaphore });
        submitInfo.setCommandBuffers({ *curFrame.commandBuffer });
        presentQueue.submit({ submitInfo }, *curFrame.inFlightFence);

        vk::PresentInfoKHR presentInfoKHR{ { *curFrame.renderFinishedSemaphore } };
        presentInfoKHR.setSwapchains({ *swapchainKHR });
        presentInfoKHR.setPImageIndices(&currentImageIdx);

        resultCheck(presentQueue.presentKHR(presentInfoKHR), "present swapchain image error");
    }

    uint32_t imageCount;
    vk::Extent2D extent;
    uint32_t currentImageIdx;
    uint32_t previousImageIdx;
    std::vector<Frame> frames;
    vk::raii::SwapchainKHR swapchainKHR;
    const vk::raii::CommandPool commandPool;
    vk::raii::CommandBuffers commandBuffers;
};

struct Shader
{
    using Stage = std::pair<const vk::ShaderStageFlagBits, const std::reference_wrapper<const std::vector<uint32_t>>>;
    Shader(const Device& device, const std::vector<Stage>& stages, const vk::PushConstantRange& pushConstantRange) :
        layout{ device, {{}, 0, {}, 1, &pushConstantRange } }
    {
        std::vector<vk::ShaderCreateInfoEXT> shaderCreateInfos{ stages.size(), { vk::ShaderCreateFlagBitsEXT::eLinkStage, {}, {}, vk::ShaderCodeTypeEXT::eSpirv, {}, {}, "main", {}, {}, 1, &pushConstantRange } };
        for(uint32_t i = 0; i < stages.size(); ++i) {
            shaderCreateInfos[i].setStage(stages[i].first);
            if(i < (stages.size() - 1)) shaderCreateInfos[i].setNextStage(stages[i + 1].first);
            shaderCreateInfos[i].setCode<uint32_t>(stages[i].second.get());
		}
        _shaders = device.device.createShadersEXT(shaderCreateInfos);
        for(const auto& shader : _shaders) shaders.emplace_back(*shader); // needed in order to pass the vector directly to bindShadersEXT()
	}
    std::vector<vk::raii::ShaderEXT> _shaders;
    std::vector<vk::ShaderEXT> shaders;
    vk::raii::PipelineLayout layout;
};

// Not used
std::string loadFile(const std::string_view path)
{
    std::ifstream in{ path.data() };
    return { (std::istreambuf_iterator{in}),std::istreambuf_iterator<char>{} };
}

int main(int /*argc*/, char* /*argv[]*/)
{
    if (!glfwInit()) exitWithError("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWmonitor* monitor = fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Vulkan Triangle Modern", monitor, nullptr);

    const vk::raii::Context context{};
    constexpr vk::ApplicationInfo applicationInfo{ nullptr, 0, nullptr, 0, vk::ApiVersion13 };

    // Instance Setup
    std::vector<const char*> iExtensions;
    uint32_t glfwInstanceExtensionCount;
    const char** glfwInstanceExtensionNames = glfwGetRequiredInstanceExtensions(&glfwInstanceExtensionCount);
    iExtensions.reserve(static_cast<size_t>(glfwInstanceExtensionCount) + 1u);
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
    vk::InstanceCreateInfo instanceCreateInfo {};
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(iExtensions);
    instanceCreateInfo.setPEnabledLayerNames(iLayers);
#ifdef __APPLE__
    instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
#endif
    vk::raii::Instance instance(context, instanceCreateInfo);

    // Surface Setup
    // unfortunately glfw surface creation does not work with the vulkan c++20 module
    vk::raii::SurfaceKHR surfaceKHR{ nullptr };
#ifdef _WIN32
    vk::Win32SurfaceCreateInfoKHR win32SurfaceCreateInfoKHR{ {}, nullptr, glfwGetWin32Window(window) };
    surfaceKHR = std::move(vk::raii::SurfaceKHR{ instance, win32SurfaceCreateInfoKHR });
#endif

    // Device setup
    vk::raii::PhysicalDevices physicalDevices{ instance };
    const vk::raii::PhysicalDevice physicalDevice{ std::move(physicalDevices[0]) };
	// * find queue
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    const auto queueFamilyIndex = findQueueFamilyIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics);
    if (!queueFamilyIndex.has_value()) exitWithError("No queue family index found");
    if (!physicalDevice.getSurfaceSupportKHR(queueFamilyIndex.value(), *surfaceKHR)) exitWithError("Queue family does not support presentation");
    // * check extensions
    std::vector dExtensions{ vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName };
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

    // Vertex buffer setup (triangle is upside down on purpose)
    std::vector vertices = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
    const size_t verticesSize = vertices.size() * sizeof(float);
    const Buffer buffer{ device, verticesSize, vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible };
    void* p = buffer.memory.mapMemory(0, vk::WholeSize);
    std::memcpy(p, vertices.data(), verticesSize);
    buffer.memory.unmapMemory();

    // Shader object setup
    // https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_object.adoc
    constexpr vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint64_t) };
    Shader shader{ device, { { vk::ShaderStageFlagBits::eVertex, vertexShaderSPV }, { vk::ShaderStageFlagBits::eFragment, fragmentShaderSPV } }, pcRange };

    // Swapchain setup
    Swapchain swapchain{ device, surfaceKHR, queueFamilyIndex.value() };
    vk::ImageMemoryBarrier2 imageMemoryBarrier{};
    imageMemoryBarrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::DependencyInfoKHR dependencyInfo{};
    dependencyInfo.setImageMemoryBarriers({ imageMemoryBarrier });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        swapchain.acquireNextImage(device);
        const auto& cFrame = swapchain.getCurrentFrame();
        const auto& cmdBuffer = cFrame.commandBuffer;

        imageMemoryBarrier.image = cFrame.image;
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        cmdBuffer.pipelineBarrier2(dependencyInfo);

        vk::RenderingAttachmentInfo rAttachmentInfo{ *cFrame.imageView, vk::ImageLayout::eColorAttachmentOptimal };
        rAttachmentInfo.clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
        rAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        rAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;

        cmdBuffer.beginRendering({ {}, { {}, swapchain.extent }, 1, 0, 1, &rAttachmentInfo });
        {
            cmdBuffer.bindShadersEXT({ vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment }, shader.shaders);
            cmdBuffer.pushConstants<uint64_t>(*shader.layout, vk::ShaderStageFlagBits::eVertex, 0, { buffer.deviceAddress });
            cmdBuffer.setPrimitiveTopologyEXT(vk::PrimitiveTopology::eTriangleList);
            cmdBuffer.setPolygonModeEXT(vk::PolygonMode::eFill);
            cmdBuffer.setFrontFaceEXT(vk::FrontFace::eCounterClockwise);
            cmdBuffer.setCullModeEXT(vk::CullModeFlagBits::eNone);
            cmdBuffer.setColorWriteMaskEXT(0, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB);
            cmdBuffer.setSampleMaskEXT(vk::SampleCountFlagBits::e1, { 0xffffffff });
            cmdBuffer.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
            cmdBuffer.setViewportWithCountEXT({ { 0, 0, static_cast<float>(swapchain.extent.width), static_cast<float>(swapchain.extent.height) } });
            cmdBuffer.setScissorWithCountEXT({ { {0, 0}, swapchain.extent } });
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
