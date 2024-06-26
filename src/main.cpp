#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <optional>
#include <algorithm>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <memory>
#include <deque>
#include <cstring>
#include "shaders.h"
import vulkan_hpp; // modules should come after all includes

#ifdef __APPLE__
constexpr bool isApple = true;
#else
constexpr bool isApple = false;
#endif

constexpr struct { uint32_t width, height; } target { 800u, 600u }; // our window
[[maybe_unused]] constexpr std::string_view shaders = R"(
[[vk::push_constant]] float3* vertices;

[shader("vertex")]
float4 vertexMain(uint vid : SV_VertexID) : SV_Position
{
    return float4(vertices[vid], 1.0);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
})";

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

struct Buffer : vk::raii::Buffer, Resource
{
    Buffer(const std::shared_ptr<Device>& device, const vk::DeviceSize size, const vk::BufferUsageFlags usageFlags, const vk::MemoryPropertyFlags memoryPropertyFlags)
        : vk::raii::Buffer{ *device, { {}, size, usageFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress } }, Resource{ device }, memory{ nullptr }
    {
        const auto memoryRequirements = getMemoryRequirements();
        const auto memoryTypeIndex = dev->findMemoryTypeIndex(memoryRequirements, memoryPropertyFlags);
        if (!memoryTypeIndex.has_value()) exitWithError("No memory type index found");
        constexpr vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        const vk::MemoryAllocateInfo memoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex.value(), &memoryAllocateFlagsInfo };
        memory = vk::raii::DeviceMemory{ *dev, memoryAllocateInfo };
        bindMemory(*memory, 0);

        const vk::BufferDeviceAddressInfo bufferDeviceAddressInfo{ **this };
        deviceAddress = dev->getBufferAddress(bufferDeviceAddressInfo); /* for bindless rendering */
    }
    vk::raii::DeviceMemory memory;
    vk::DeviceAddress deviceAddress;
};

struct Swapchain : Resource
{
    // Data for one frame/image in our swapchain, recreated every frame
    struct Frame {
        Frame(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool) :
    		presentFinishFence{ device, vk::FenceCreateInfo{} }, imageAvailableSemaphore{ device, vk::SemaphoreCreateInfo{} }, renderFinishedSemaphore{ device, vk::SemaphoreCreateInfo{} },
    		commandBuffer{ std::move(vk::raii::CommandBuffers{ device, { *commandPool, vk::CommandBufferLevel::ePrimary, 1 } }[0]) }
        {}
        vk::raii::Fence presentFinishFence;
        vk::raii::Semaphore imageAvailableSemaphore, renderFinishedSemaphore;
        vk::raii::CommandBuffer commandBuffer;
    };

    Swapchain(const std::shared_ptr<Device>& device, const vk::raii::SurfaceKHR& surface, const uint32_t queueFamilyIndex) : Resource{ device }, currentImageIdx{ 0 }, previousImageIdx{ 0 },
		swapchainKHR{ nullptr }, commandPool{ *dev, { vk::CommandPoolCreateFlagBits::eTransient, queueFamilyIndex } }
    {
        const auto surfaceCapabilities = dev->physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const auto surfaceFormats = dev->physicalDevice.getSurfaceFormatsKHR(*surface);

        imageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if (surfaceCapabilities.maxImageCount) imageCount = std::min(imageCount, surfaceCapabilities.maxImageCount);
        currentImageIdx = imageCount - 1u; // for init
        extent = surfaceCapabilities.currentExtent; format = surfaceFormats[0].format;
        vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR{ { vk::SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocationEXT },
    		*surface, imageCount, format, surfaceFormats[0].colorSpace,
    		extent,1u, vk::ImageUsageFlagBits::eColorAttachment };
        swapchainCreateInfoKHR.setPresentMode(vk::PresentModeKHR::eFifo);
        swapchainKHR = vk::raii::SwapchainKHR{ *dev, swapchainCreateInfoKHR };

        images = swapchainKHR.getImages();
        views.reserve(images.size());
        for (const auto& image : images) views.emplace_back(nullptr);
    }

    Frame& acquireNewFrame() {
        for (auto it = frames.begin(); it != frames.end(); (it->presentFinishFence.getStatus() == vk::Result::eSuccess) ? it = frames.erase(it) : ++it) {}
        frames.emplace_back(*dev, commandPool); // create a new frame
        return frames.back();
    }

    void acquireNextImage() {
        auto& frame = acquireNewFrame();
        auto [result, idx] = swapchainKHR.acquireNextImage(UINT64_MAX, *frame.imageAvailableSemaphore);
        if (result != vk::Result::eSuccess) exitWithError("acquiring next image error");
        currentImageIdx = idx;
        /* create image view after image is acquired because of vk::SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocationEXT */
        if(not *views[currentImageIdx]) {
        	views[currentImageIdx] = vk::raii::ImageView{ *dev, vk::ImageViewCreateInfo{ {}, images[currentImageIdx], vk::ImageViewType::e2D,
                format, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } } };
        }
        frame.commandBuffer.begin({});
    }

    void submitImage(const vk::raii::Queue& presentQueue) {
        auto& frame = frames.back();
        frame.commandBuffer.end();

        constexpr vk::PipelineStageFlags waitDstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        presentQueue.submit(vk::SubmitInfo{ *frame.imageAvailableSemaphore, 
            waitDstStageMask, *frame.commandBuffer, *frame.renderFinishedSemaphore });
        vk::SwapchainPresentFenceInfoEXT presentFenceInfo{ *frame.presentFinishFence };
        if (presentQueue.presentKHR({ *frame.renderFinishedSemaphore, *swapchainKHR, currentImageIdx, {}, &presentFenceInfo }) != vk::Result::eSuccess) exitWithError("present swapchain image error");
    }

    Frame& getCurrentFrame() { return frames.back(); }
    vk::Image& getCurrentImage() { return images[currentImageIdx]; }
    vk::raii::ImageView& getCurrentImageView() { return views[currentImageIdx]; }

    vk::Extent2D extent;
    vk::Format format;
    uint32_t imageCount, currentImageIdx, previousImageIdx;
    vk::raii::SwapchainKHR swapchainKHR;
    std::vector<vk::Image> images;
    std::vector<vk::raii::ImageView> views;
    vk::raii::CommandPool commandPool;
    std::deque<Frame> frames;
};

struct Shader : Resource
{
    struct Stage {
        Stage(const vk::ShaderStageFlagBits stage, const std::reference_wrapper<const std::vector<uint32_t>> spv, std::string entry = "main") : stage{ stage }, spv{ spv }, entry{std::move(entry)} {}
		vk::ShaderStageFlagBits stage; std::reference_wrapper<const std::vector<uint32_t>> spv; std::string entry;
	};
    Shader(const std::shared_ptr<Device>& device, const std::vector<Stage>& shaderStages, const std::vector<vk::PushConstantRange>& pcRanges) : Resource{ device },
        shaders{ shaderStages.size(), nullptr }, stages{ shaderStages.size() }, layout{ *dev, vk::PipelineLayoutCreateInfo{}.setPushConstantRanges(pcRanges) }
    {
        std::vector shaderCreateInfos{ shaderStages.size(), vk::ShaderCreateInfoEXT{ shaderStages.size() > 1u ? vk::ShaderCreateFlagBitsEXT::eLinkStage : vk::ShaderCreateFlagsEXT{} }
            .setCodeType(vk::ShaderCodeTypeEXT::eSpirv).setPushConstantRanges(pcRanges) };
        for (size_t i = 0; i < shaderStages.size(); ++i) {
            shaderCreateInfos[i].setStage(shaderStages[i].stage).setPName(shaderStages[i].entry.c_str());
            if (i < (shaderStages.size() - 1)) shaderCreateInfos[i].setNextStage(shaderStages[i + 1u].stage);
            shaderCreateInfos[i].setCode<uint32_t>(shaderStages[i].spv.get());
            stages[i] = shaderStages[i].stage;
        }
        _shaders = dev->createShadersEXT(shaderCreateInfos);
        for (size_t i = 0; i < shaderStages.size(); ++i) shaders[i] = *_shaders[i]; // needed in order to pass the vector directly to bindShadersEXT()
    }
    std::vector<vk::raii::ShaderEXT> _shaders;
    std::vector<vk::ShaderEXT> shaders;
    std::vector<vk::ShaderStageFlagBits> stages;
    vk::raii::PipelineLayout layout;
};

int main(int /*argc*/, char** /*argv*/)
{
    if (!glfwInit()) exitWithError("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(target.width, target.height, "Vulkan Triangle Modern", nullptr, nullptr);

    const vk::raii::Context context;
    // Instance Setup
    std::vector iExtensions{ vk::KHRSurfaceExtensionName, vk::EXTSurfaceMaintenance1ExtensionName, vk::KHRGetSurfaceCapabilities2ExtensionName };
#ifdef VK_USE_PLATFORM_WIN32_KHR
    iExtensions.emplace_back(vk::KHRWin32SurfaceExtensionName);
#elif VK_USE_PLATFORM_XLIB_KHR
    iExtensions.emplace_back(vk::KHRXlibSurfaceExtensionName);
#elif VK_USE_PLATFORM_WAYLAND_KHR
    iExtensions.emplace_back(vk::KHRWaylandSurfaceExtensionName);
#elif VK_USE_PLATFORM_METAL_EXT
    iExtensions.emplace_back(vk::EXTMetalSurfaceExtensionName);
    iExtensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
#endif

    std::vector iLayers = { "VK_LAYER_LUNARG_monitor" };
#if !defined( NDEBUG )
    iLayers.emplace_back("VK_LAYER_KHRONOS_validation");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) iLayers.clear();
#endif
    iLayers.emplace_back("VK_LAYER_KHRONOS_shader_object"); // always activate this layer since everyone except NVIDIA requires it for now
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) exitWithError("Instance layers not available");
    if (!extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions)) exitWithError("Instance extensions not available");

    constexpr vk::ApplicationInfo applicationInfo{ nullptr, 0, nullptr, 0, vk::ApiVersion13 };
    vk::InstanceCreateInfo instanceCreateInfo{ {}, &applicationInfo, iLayers, iExtensions };
    if constexpr (isApple) instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
    const vk::raii::Instance instance(context, instanceCreateInfo);

    // Surface Setup
    // unfortunately glfw surface creation does not work with the vulkan c++20 module
#ifdef VK_USE_PLATFORM_WIN32_KHR
    vk::raii::SurfaceKHR surfaceKHR { instance, vk::Win32SurfaceCreateInfoKHR{ {}, GetModuleHandle(nullptr), glfwGetWin32Window(window) } };
#elif VK_USE_PLATFORM_XLIB_KHR
    vk::raii::SurfaceKHR surfaceKHR { instance, vk::XlibSurfaceCreateInfoKHR{ {}, glfwGetX11Display(), glfwGetX11Window(window) } };
#elif VK_USE_PLATFORM_WAYLAND_KHR
    vk::raii::SurfaceKHR surfaceKHR { instance, vk::WaylandSurfaceCreateInfoKHR{ {}, glfwGetWaylandDisplay(), glfwGetWaylandWindow(window) } };
#elif VK_USE_PLATFORM_METAL_EXT
    exitWithError("Metal surface not implemented");
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
    std::vector dExtensions{ vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName, vk::EXTSwapchainMaintenance1ExtensionName };
    if constexpr (isApple) dExtensions.emplace_back("VK_KHR_portability_subset");
    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
    // * activate features
    auto vulkan11Features = vk::PhysicalDeviceVulkan11Features{}.setVariablePointers(true).setVariablePointersStorageBuffer(true);
    auto bufferDeviceAddressFeatures = vk::PhysicalDeviceBufferDeviceAddressFeatures{ true }.setPNext(&vulkan11Features);
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance{ true, &bufferDeviceAddressFeatures };
    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ true, &swapchainMaintenance };
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

    // Shader object setup : https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_object.adoc
    constexpr vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint64_t) };
    Shader shader{ device, { { vk::ShaderStageFlagBits::eVertex, shaders_spv, "vertexMain" }, { vk::ShaderStageFlagBits::eFragment, shaders_spv, "fragmentMain" } }, { pcRange } };

    // Swapchain setup
    Swapchain swapchain{ device, surfaceKHR, queueFamilyIndex.value() };
    vk::ImageMemoryBarrier2 imageMemoryBarrier {};
    imageMemoryBarrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(imageMemoryBarrier);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) continue;
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        swapchain.acquireNextImage();
        const auto& cFrame = swapchain.getCurrentFrame();
        const auto& cmdBuffer = cFrame.commandBuffer;

        imageMemoryBarrier.image = swapchain.getCurrentImage();
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        cmdBuffer.pipelineBarrier2(dependencyInfo);
        
        vk::RenderingAttachmentInfo rAttachmentInfo{ *swapchain.getCurrentImageView(), vk::ImageLayout::eColorAttachmentOptimal};
        rAttachmentInfo.clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
        rAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        rAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        cmdBuffer.beginRendering({ {}, { {}, swapchain.extent }, 1, 0, 1, &rAttachmentInfo });
        {
            /* set render state for shader objects */
            cmdBuffer.bindShadersEXT(shader.stages, shader.shaders);
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
            cmdBuffer.setColorBlendEnableEXT(0, { false });
            cmdBuffer.setDepthTestEnableEXT(false);
            cmdBuffer.setDepthWriteEnableEXT(false);
            cmdBuffer.setDepthBiasEnableEXT(false);
            cmdBuffer.setStencilTestEnableEXT(false);
            cmdBuffer.setRasterizerDiscardEnableEXT(false);
            cmdBuffer.setColorBlendEquationEXT(0, vk::ColorBlendEquationEXT{}.setSrcColorBlendFactor(vk::BlendFactor::eOne));
            cmdBuffer.setAlphaToCoverageEnableEXT(false);
            cmdBuffer.setPrimitiveRestartEnableEXT(false);
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
