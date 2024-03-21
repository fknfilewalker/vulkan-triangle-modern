#ifdef __APPLE__
#include <vulkan/vulkan_raii.hpp>
constexpr bool isApple = true;
#else
import vulkan_hpp;
constexpr bool isApple = false;
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <optional>
#include <algorithm>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <memory>

constexpr bool fullscreen = false;
constexpr uint32_t window_width = 800;
constexpr uint32_t window_height = 600;
[[maybe_unused]] constexpr std::string_view vertexShader = R"(
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
	uint64_t vertexPtr; /* for bindless rendering */
};

void main() {
	Vertex vertex = Vertex(vertexPtr + sizeOfVec3 * gl_VertexIndex);
	gl_Position = vec4(vertex.position.xy, 0.0, 1.0);
})";
[[maybe_unused]] constexpr std::string_view fragmentShader = R"(
#version 450
layout (location = 0) out vec4 fragColor;
void main() {
	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
})";

#include "shaders.h"

[[noreturn]] void exitWithError(const std::string_view error) {
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

struct Device : vk::raii::Device
{
    using QueueFamily = uint32_t;
    using QueueCount = uint32_t;
    using Queues = std::unordered_map<QueueFamily, QueueCount>;
    Device(const vk::raii::PhysicalDevice& physicalDevice, const std::vector<const char*>& extensions, const Queues& queues, const void* pNext) :
        vk::raii::Device{ nullptr }, physicalDevice{ physicalDevice }, memoryProperties{ physicalDevice.getMemoryProperties() }
    {
        constexpr float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
        deviceQueueCreateInfos.reserve(queues.size());
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
            deviceQueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queueCount, &priority });
        }
        const vk::DeviceCreateInfo deviceCreateInfo{ {}, deviceQueueCreateInfos, {}, extensions,{}, pNext };
        vk::raii::Device::operator=({ physicalDevice, deviceCreateInfo });
        // get all our queues -> queue[family][index]
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
            queue.emplace_back(std::vector<vk::raii::Queue>{ queueCount, nullptr });
            for (uint32_t i = 0; i < queueCount; ++i) queue.back()[i] = getQueue(queueFamilyIndex, i);
        }
    }

    [[nodiscard]] std::optional<uint32_t> findMemoryTypeIndex(const vk::MemoryRequirements& requirements, const vk::MemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
        return std::nullopt;
    }

    operator const vk::raii::PhysicalDevice& () const { return physicalDevice; }

    std::vector<std::vector<vk::raii::Queue>> queue;
    vk::raii::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceMemoryProperties memoryProperties;
};

// Every resource has a device reference
struct Resource { std::shared_ptr<Device> dev; };

struct Buffer : Resource
{
    Buffer(const std::shared_ptr<Device>& device, const vk::DeviceSize size, const vk::BufferUsageFlags usageFlags, const vk::MemoryPropertyFlags memoryPropertyFlags)
        : Resource{ device }, buffer{ *dev, { {}, size, usageFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress } }, memory{ nullptr }
    {
        const auto memoryRequirements = buffer.getMemoryRequirements();
        const auto memoryTypeIndex = dev->findMemoryTypeIndex(memoryRequirements, memoryPropertyFlags);
        if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");
        constexpr vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        const vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value(), &memoryAllocateFlagsInfo };
        memory = vk::raii::DeviceMemory{ *dev, memoryAllocateInfo };
        buffer.bindMemory(*memory, 0);

        const vk::BufferDeviceAddressInfo bufferDeviceAddressInfo{ *buffer };
        deviceAddress = dev->getBufferAddress(bufferDeviceAddressInfo); /* for bindless rendering */
    }
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    vk::DeviceAddress deviceAddress;
};

struct Swapchain : Resource
{
    // Data for one frame/image in our swapchain
    struct Frame {
        Frame(const vk::raii::Device& device, const vk::Image& image, const vk::Format format, vk::raii::CommandBuffer& commandBuffer) : image{ image }, imageView{ nullptr },
            inFlightFence{ device, vk::FenceCreateInfo{ vk::FenceCreateFlagBits::eSignaled } }, nextImageAvailableSemaphore{ device, vk::SemaphoreCreateInfo{} },
            renderFinishedSemaphore{ device, vk::SemaphoreCreateInfo{} }, commandBuffer{ std::move(commandBuffer) }
        {
            imageView = vk::raii::ImageView{ device, vk::ImageViewCreateInfo{ {}, image, vk::ImageViewType::e2D, format,
                {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } } };
        }
        vk::Image image;
        vk::raii::ImageView imageView;
        vk::raii::Fence inFlightFence;
        vk::raii::Semaphore nextImageAvailableSemaphore /* refers to the next frame */, renderFinishedSemaphore /* refers to current frame */;
        vk::raii::CommandBuffer commandBuffer;
    };

    Swapchain(const std::shared_ptr<Device>& device, const vk::raii::SurfaceKHR& surface, const uint32_t queueFamilyIndex) : Resource{ device }, currentImageIdx{ 0 },
		previousImageIdx{ 0 }, swapchainKHR{ nullptr }, commandPool{ *dev, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex } }
    {
        const auto surfaceCapabilities = dev->physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const auto surfaceFormats = dev->physicalDevice.getSurfaceFormatsKHR(*surface);

        imageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if (surfaceCapabilities.maxImageCount) imageCount = std::min(imageCount, surfaceCapabilities.maxImageCount);
        currentImageIdx = imageCount - 1u; // just for init
        extent = surfaceCapabilities.currentExtent;
        const vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR{ {}, *surface, imageCount,
            surfaceFormats[0].format, surfaceFormats[0].colorSpace, extent, 1u, vk::ImageUsageFlagBits::eColorAttachment };
        swapchainKHR = vk::raii::SwapchainKHR{ *dev, swapchainCreateInfoKHR };

        auto commandBuffers = vk::raii::CommandBuffers{ *dev, { *commandPool, vk::CommandBufferLevel::ePrimary, imageCount } };
        const std::vector<vk::Image> images = swapchainKHR.getImages();
        frames.reserve(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) frames.emplace_back(*dev, images[i], surfaceFormats[0].format, commandBuffers[i]);
    }
    ~Swapchain() { for (auto& frame : frames) resultCheck(dev->waitForFences(*frame.inFlightFence, vk::True, UINT64_MAX), "waiting for fence error"); }

    void acquireNextImage() {
        const Frame& oldFrame = getCurrentFrame();
        resultCheck(dev->waitForFences(*oldFrame.inFlightFence, vk::True, UINT64_MAX), "waiting for fence error");
        dev->resetFences(*oldFrame.inFlightFence);
        const std::pair<vk::Result, uint32_t> nextImage = swapchainKHR.acquireNextImage(0, *oldFrame.nextImageAvailableSemaphore, *oldFrame.inFlightFence);
        resultCheck(nextImage.first, "acquiring next swapchain image error");
        previousImageIdx = currentImageIdx;
        currentImageIdx = nextImage.second;

        const Frame& newFrame = getCurrentFrame();
        newFrame.commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }

    void submitImage(const vk::raii::Queue& presentQueue) {
        const Frame& curFrame = getCurrentFrame();
        curFrame.commandBuffer.end();

        constexpr vk::PipelineStageFlags waitDstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        presentQueue.submit(vk::SubmitInfo{ *getPreviousFrame().nextImageAvailableSemaphore, waitDstStageMask,
            *curFrame.commandBuffer, *curFrame.renderFinishedSemaphore });
        resultCheck(presentQueue.presentKHR({ *curFrame.renderFinishedSemaphore, *swapchainKHR, currentImageIdx }), "present swapchain image error");
    }

    const Frame& getCurrentFrame() { return frames[currentImageIdx]; }
    const Frame& getPreviousFrame() { return frames[previousImageIdx]; }

    vk::Extent2D extent;
    uint32_t imageCount, currentImageIdx, previousImageIdx;
    vk::raii::SwapchainKHR swapchainKHR;
    vk::raii::CommandPool commandPool;
    std::vector<Frame> frames;
};

struct Shader : Resource
{
    using Stage = std::pair<const vk::ShaderStageFlagBits, const std::reference_wrapper<const std::vector<uint32_t>>>;
    Shader(const std::shared_ptr<Device>& device, const std::vector<Stage>& stages, const std::vector<vk::PushConstantRange>& pcRanges) : Resource{ device },
        layout{ *dev, vk::PipelineLayoutCreateInfo{}.setPushConstantRanges(pcRanges) }
    {
        std::vector shaderCreateInfos{ stages.size(), vk::ShaderCreateInfoEXT{ stages.size() > 1u ? vk::ShaderCreateFlagBitsEXT::eLinkStage : vk::ShaderCreateFlagsEXT{} }
        	.setCodeType(vk::ShaderCodeTypeEXT::eSpirv).setPName("main").setPushConstantRanges(pcRanges) };
        for (size_t i = 0; i < stages.size(); ++i) {
            shaderCreateInfos[i].setStage(stages[i].first);
            if (i < (stages.size() - 1)) shaderCreateInfos[i].setNextStage(stages[i + 1u].first);
            shaderCreateInfos[i].setCode<uint32_t>(stages[i].second.get());
        }
        _shaders = dev->createShadersEXT(shaderCreateInfos);
        for (const auto& shader : _shaders) shaders.emplace_back(*shader); // needed in order to pass the vector directly to bindShadersEXT()
    }
    std::vector<vk::raii::ShaderEXT> _shaders;
    std::vector<vk::ShaderEXT> shaders;
    vk::raii::PipelineLayout layout;
};

int main(int /*argc*/, char** /*argv*/)
{
    if (!glfwInit()) exitWithError("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWmonitor* monitor = fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Vulkan Triangle Modern", monitor, nullptr);

    const vk::raii::Context context{};
    constexpr vk::ApplicationInfo applicationInfo{ nullptr, 0, nullptr, 0, vk::ApiVersion12 };

    // Instance Setup
    std::vector<const char*> iExtensions;
    uint32_t glfwInstanceExtensionCount;
    const char** glfwInstanceExtensionNames = glfwGetRequiredInstanceExtensions(&glfwInstanceExtensionCount);
    iExtensions.reserve(static_cast<size_t>(glfwInstanceExtensionCount) + 1u);
    for (uint32_t i = 0; i < glfwInstanceExtensionCount; ++i) iExtensions.emplace_back(glfwInstanceExtensionNames[i]);
    if constexpr (isApple) iExtensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
    
	std::vector iLayers = { "VK_LAYER_LUNARG_monitor" };
#if !defined( NDEBUG )
    iLayers.emplace_back("VK_LAYER_KHRONOS_validation");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) iLayers.clear();
#endif
    if constexpr (isApple) iLayers.emplace_back("VK_LAYER_KHRONOS_shader_object");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) exitWithError("Instance layers not available");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions)) exitWithError("Instance extensions not available");

    vk::InstanceCreateInfo instanceCreateInfo{ {}, &applicationInfo, iLayers, iExtensions };
	if constexpr (isApple) instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
    const vk::raii::Instance instance(context, instanceCreateInfo);

    // Surface Setup
    // unfortunately glfw surface creation does not work with the vulkan c++20 module
#ifdef _WIN32
    vk::raii::SurfaceKHR surfaceKHR { instance, vk::Win32SurfaceCreateInfoKHR{ {}, GetModuleHandle(nullptr), glfwGetWin32Window(window) } };
#elif __APPLE__
    VkSurfaceKHR _surface;
    glfwCreateWindowSurface(*instance, window, nullptr, &_surface);
    vk::raii::SurfaceKHR surfaceKHR { instance, _surface };
#endif
    // Device setup
    const vk::raii::PhysicalDevices physicalDevices{ instance };
    const vk::raii::PhysicalDevice& physicalDevice{ physicalDevices[0] };
    // * find queue
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    const auto queueFamilyIndex = findQueueFamilyIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics);
    if (!queueFamilyIndex.has_value()) exitWithError("No queue family index found");
    if (!physicalDevice.getSurfaceSupportKHR(queueFamilyIndex.value(), *surfaceKHR)) exitWithError("Queue family does not support presentation");
    // * check extensions
    std::vector dExtensions{ vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName, vk::KHRDynamicRenderingExtensionName, vk::KHRSynchronization2ExtensionName };
    if constexpr (isApple) dExtensions.emplace_back("VK_KHR_portability_subset");

    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
    // * activate features
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ true };
    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ true, &bufferDeviceAddressFeatures };
    vk::PhysicalDeviceSynchronization2Features synchronization2Features{ true, &shaderObjectFeatures };
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{ true, &synchronization2Features };
    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2{ {}, &dynamicRenderingFeatures };
    physicalDeviceFeatures2.features.shaderInt64 = true;
    // * create device
    auto device = std::make_shared<Device>(physicalDevice, dExtensions, Device::Queues{{queueFamilyIndex.value(), 1}}, &physicalDeviceFeatures2);

    // Vertex buffer setup (triangle is upside down on purpose)
    const std::vector vertices = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
    const size_t verticesSize = vertices.size() * sizeof(float);
    const Buffer buffer{ device, verticesSize, vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible }; /* reBAR */
    void* p = buffer.memory.mapMemory(0, vk::WholeSize);
    std::memcpy(p, vertices.data(), verticesSize);
    buffer.memory.unmapMemory();

    // Shader object setup
    // https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_object.adoc
    constexpr vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint64_t) };
    Shader shader{ device, { { vk::ShaderStageFlagBits::eVertex, vertexShaderSPV }, { vk::ShaderStageFlagBits::eFragment, fragmentShaderSPV } }, { pcRange } };

    // Swapchain setup
    Swapchain swapchain{ device, surfaceKHR, queueFamilyIndex.value() };
    vk::ImageMemoryBarrier2 imageMemoryBarrier {};
    imageMemoryBarrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(imageMemoryBarrier);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        swapchain.acquireNextImage();
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
            /* set render state for shader objects */
            cmdBuffer.bindShadersEXT({ vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment }, shader.shaders);
            cmdBuffer.pushConstants<uint64_t>(*shader.layout, vk::ShaderStageFlagBits::eVertex, 0, /* for bindless rendering */ buffer.deviceAddress);
            cmdBuffer.setPrimitiveTopologyEXT(vk::PrimitiveTopology::eTriangleList);
            cmdBuffer.setPolygonModeEXT(vk::PolygonMode::eFill);
            cmdBuffer.setFrontFaceEXT(vk::FrontFace::eCounterClockwise);
            cmdBuffer.setCullModeEXT(vk::CullModeFlagBits::eNone);
            cmdBuffer.setColorWriteMaskEXT(0, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB);
            cmdBuffer.setSampleMaskEXT(vk::SampleCountFlagBits::e1, { 0xffffffff });
            cmdBuffer.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
            cmdBuffer.setViewportWithCountEXT({ { 0, 0, static_cast<float>(swapchain.extent.width), static_cast<float>(swapchain.extent.height) } });
            cmdBuffer.setScissorWithCountEXT({ { { 0, 0 }, swapchain.extent } });
            cmdBuffer.setVertexInputEXT({}, {});
            cmdBuffer.setColorBlendEnableEXT(0, vk::False);
            cmdBuffer.setDepthTestEnableEXT(vk::False);
            cmdBuffer.setDepthWriteEnableEXT(vk::False);
            cmdBuffer.setDepthBiasEnableEXT(vk::False);
            cmdBuffer.setStencilTestEnableEXT(vk::False);
            cmdBuffer.setRasterizerDiscardEnableEXT(vk::False);
            cmdBuffer.setColorBlendEquationEXT(0, vk::ColorBlendEquationEXT{}.setSrcColorBlendFactor(vk::BlendFactor::eOne));
            cmdBuffer.setAlphaToCoverageEnableEXT(vk::False);
            cmdBuffer.setPrimitiveRestartEnableEXT(vk::False);
            cmdBuffer.draw(3, 1, 0, 0);
        }
        cmdBuffer.endRendering();

        imageMemoryBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        imageMemoryBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        cmdBuffer.pipelineBarrier2(dependencyInfo);
        swapchain.submitImage(device->queue[queueFamilyIndex.value()][0]);
    }
    device->waitIdle();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
