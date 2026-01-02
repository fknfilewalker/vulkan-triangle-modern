#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <optional>
#include <algorithm>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <memory>
#include <deque>
#include <cstring>
#include <span>
#include <limits>
#include "shaders.h"
import vulkan_hpp; // modules should come after all includes

#ifdef __APPLE__
constexpr bool isApple = true;
#else
constexpr bool isApple = false;
#endif

constexpr struct { uint32_t width, height; } target { 800u, 600u }; // our window
[[maybe_unused]] constexpr std::string_view shaders = R"(
[vk::push_constant] float4* vertices;

struct OutIn
{
    float4 position : SV_POSITION, color : COLOR;
};

[shader("vertex")]
OutIn vertexMain(uint vid : SV_VertexID)
{
    return { vertices[vid], { float3(uint3(0, 1, 2) == (vid % 3)), 1.0 } };
}

[shader("fragment")]
float4 fragmentMain(OutIn input) : SV_Target
{
    return input.color;
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
            if constexpr (std::is_same_v<vk::LayerProperties, T>) return std::string_view{ availableElement.layerName.data() } == requestedElement;
            else if constexpr (std::is_same_v<vk::ExtensionProperties, T>) return std::string_view{ availableElement.extensionName.data() } == requestedElement;
        }) != available.end();
    });
}

std::optional<uint32_t> findQueueFamilyIndex(const vk::raii::PhysicalDevice& physicalDevice, const vk::QueueFlags queueFlags, const vk::Instance instance = nullptr) {
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    std::optional<uint32_t> bestFamily;
    std::bitset<12> bestScore = 0;
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        // check if queue family supports all requested queue flags
        if (static_cast<uint32_t>(queueFamilyProperties[i].queueFlags & queueFlags) == static_cast<uint32_t>(queueFlags)) {
            const std::bitset<12> score = static_cast<uint32_t>(queueFamilyProperties[i].queueFlags);
            // use queue family with the least other bits set
            if (!bestFamily.has_value() || score.count() < bestScore.count()) {
                // check presentation support too if instance is given
                if(instance == nullptr || SDL_Vulkan_GetPresentationSupport(instance, *physicalDevice, i)) {
                    bestFamily = i;
                    bestScore = score;
                }
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
            queue.emplace_back( queueCount, nullptr );
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
        swapchain{ nullptr }, commandPool{ *dev, { vk::CommandPoolCreateFlagBits::eTransient, queueFamilyIndex } }
    {
        const auto surfaceCapabilities = dev->physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const auto surfaceFormats = dev->physicalDevice.getSurfaceFormatsKHR(*surface);

        imageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if (surfaceCapabilities.maxImageCount) imageCount = std::min(imageCount, surfaceCapabilities.maxImageCount);
        swapchainCreateInfo = vk::SwapchainCreateInfoKHR{ { /* vk::SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocation */ }, // causes problems with apps like RiverTuner
    		*surface, imageCount, surfaceFormats[0].format, surfaceFormats[0].colorSpace, surfaceCapabilities.currentExtent,
        	1u, vk::ImageUsageFlagBits::eColorAttachment }.setPresentMode(vk::PresentModeKHR::eImmediate);
        createSwapchain();
    }

    void createSwapchain(uint32_t width = 0, uint32_t height = 0) {
        const auto sc = dev->physicalDevice.getSurfaceCapabilitiesKHR(swapchainCreateInfo.surface);
        const bool valid = sc.currentExtent.width != std::numeric_limits<uint32_t>::max();
        swapchainCreateInfo.imageExtent.width = valid ? sc.currentExtent.width : std::clamp(width, sc.minImageExtent.width, sc.maxImageExtent.width);
        swapchainCreateInfo.imageExtent.height = valid ? sc.currentExtent.height : std::clamp(height, sc.minImageExtent.height, sc.maxImageExtent.height);
        swapchainCreateInfo.oldSwapchain = *swapchain;
        swapchain = vk::raii::SwapchainKHR{ *dev, swapchainCreateInfo };
        images = swapchain.getImages();
        views.clear(); for (const auto& image : images) views.emplace_back(nullptr);
    }

    Frame& acquireNewFrame() {
        for (auto it = frames.begin(); it != frames.end(); (it->presentFinishFence.getStatus() == vk::Result::eSuccess) ? it = frames.erase(it) : ++it) {}
        frames.emplace_back(*dev, commandPool); // create a new frame
        return frames.back();
    }

    void acquireNextImage() {
        auto& frame = acquireNewFrame();
        currentImageIdx = swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailableSemaphore).value;
        /* create image view after image is acquired because of vk::SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocation */
        if(not *views[currentImageIdx]) {
        	views[currentImageIdx] = vk::raii::ImageView{ *dev, vk::ImageViewCreateInfo{ {}, images[currentImageIdx], vk::ImageViewType::e2D,
                swapchainCreateInfo.imageFormat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } } };
        }
        frame.commandBuffer.begin({});
    }

    void submitImage(const vk::raii::Queue& presentQueue, const vk::PipelineStageFlags waitDstStageMask) {
        const auto& frame = frames.back();
        frame.commandBuffer.end();
        presentQueue.submit(vk::SubmitInfo{ *frame.imageAvailableSemaphore, 
            waitDstStageMask, *frame.commandBuffer, *frame.renderFinishedSemaphore });
        const vk::SwapchainPresentFenceInfoKHR presentFenceInfo{ *frame.presentFinishFence };
        auto _ = presentQueue.presentKHR({ *frame.renderFinishedSemaphore, *swapchain, currentImageIdx, {}, &presentFenceInfo });
    }

    Frame& getCurrentFrame() { return frames.back(); }
    vk::Image& getCurrentImage() { return images[currentImageIdx]; }
    vk::raii::ImageView& getCurrentImageView() { return views[currentImageIdx]; }
    [[nodiscard]] const vk::Extent2D& extent() const { return swapchainCreateInfo.imageExtent; }

    vk::SwapchainCreateInfoKHR swapchainCreateInfo;
    uint32_t imageCount, currentImageIdx, previousImageIdx;
    vk::raii::SwapchainKHR swapchain;
    std::vector<vk::Image> images;
    std::vector<vk::raii::ImageView> views;
    vk::raii::CommandPool commandPool;
    std::deque<Frame> frames;
};

struct Shader : Resource
{
	using Stage = std::tuple<const vk::ShaderStageFlagBits, const std::span<uint32_t const>, std::string_view>; // stage, spv, entry
    Shader(const std::shared_ptr<Device>& device, const std::vector<Stage>& shaderStages, const std::vector<vk::PushConstantRange>& pcRanges) : Resource{ device },
        shaders{ shaderStages.size(), nullptr }, stages{ shaderStages.size() }, layout{ *dev, vk::PipelineLayoutCreateInfo{}.setPushConstantRanges(pcRanges) }
    {
        std::vector shaderCreateInfos{ shaderStages.size(), vk::ShaderCreateInfoEXT{ shaderStages.size() > 1u ? vk::ShaderCreateFlagBitsEXT::eLinkStage : vk::ShaderCreateFlagsEXT{} }
            .setCodeType(vk::ShaderCodeTypeEXT::eSpirv).setPushConstantRanges(pcRanges) };
        for (size_t i = 0; i < shaderStages.size(); ++i) {
            shaderCreateInfos[i].setStage(std::get<0>(shaderStages[i])).setPName(std::get<2>(shaderStages[i]).data());
            if (i < (shaderStages.size() - 1)) shaderCreateInfos[i].setNextStage(std::get<0>(shaderStages[i + 1u]));
			shaderCreateInfos[i].setCode<uint32_t>(std::get<1>(shaderStages[i]));
			stages[i] = std::get<0>(shaderStages[i]);
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
    if (!SDL_Init(0)) exitWithError("Failed to init SDL");
    SDL_Window* window = SDL_CreateWindow("Vulkan Triangle Modern", target.width, target.height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    const vk::raii::Context context;
    // Instance Setup
    std::vector iExtensions{ vk::KHRSurfaceMaintenance1ExtensionName, vk::KHRGetSurfaceCapabilities2ExtensionName };
    {
        uint32_t count;
        const auto sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&count);
        iExtensions.insert(iExtensions.end(), sdlExtensions, sdlExtensions + count);
    }
    if constexpr (isApple) iExtensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);

    std::vector iLayers = { "VK_LAYER_LUNARG_monitor" };
#if !defined( NDEBUG )
    iLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) iLayers.clear();
    iLayers.emplace_back("VK_LAYER_KHRONOS_shader_object"); // always try to activate this layer since many drivers still require this (requires Vulkan SDK though)
    if (!extensionsOrLayersAvailable(context.enumerateInstanceLayerProperties(), iLayers)) iLayers.clear();
    if (!extensionsOrLayersAvailable(context.enumerateInstanceExtensionProperties(), iExtensions)) exitWithError("Instance extensions not available");

    constexpr vk::ApplicationInfo applicationInfo{ nullptr, 0, nullptr, 0, vk::ApiVersion13 };
    vk::InstanceCreateInfo instanceCreateInfo{ {}, &applicationInfo, iLayers, iExtensions };
    if constexpr (isApple) instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
    const vk::raii::Instance instance(context, instanceCreateInfo);

    // Surface Setup
    vk::raii::SurfaceKHR surface{ instance, nullptr };
    if (!SDL_Vulkan_CreateSurface(window, *instance, nullptr, (VkSurfaceKHR*)&*surface)) exitWithError("Failed to create Vulkan surface");
    // Device setup
    const vk::raii::PhysicalDevices physicalDevices{ instance };
    const vk::raii::PhysicalDevice& physicalDevice{ physicalDevices[0] };
    // * find queue
    const auto queueFamilyIndex = findQueueFamilyIndex(physicalDevice, vk::QueueFlagBits::eGraphics, instance);
    if (!queueFamilyIndex.has_value()) exitWithError("No queue family index found");
    // * check extensions
    std::vector dExtensions{ vk::KHRSwapchainExtensionName, vk::EXTShaderObjectExtensionName };
    if constexpr (isApple) dExtensions.emplace_back(vk::EXTSwapchainMaintenance1ExtensionName);
    else dExtensions.emplace_back(vk::KHRSwapchainMaintenance1ExtensionName);
    if constexpr (isApple) dExtensions.emplace_back("VK_KHR_portability_subset");
    if (!extensionsOrLayersAvailable(physicalDevice.enumerateDeviceExtensionProperties(), dExtensions)) exitWithError("Device extensions not available");
    // * activate features
    auto vulkan13Features = vk::PhysicalDeviceVulkan13Features{}.setDynamicRendering(true).setSynchronization2(true);
    auto vulkan12Features = vk::PhysicalDeviceVulkan12Features{}.setBufferDeviceAddress(true).setPNext(vulkan13Features);
    auto vulkan11Features = vk::PhysicalDeviceVulkan11Features{}.setShaderDrawParameters(true).setPNext(vulkan12Features);
    auto shaderObjectFeatures = vk::PhysicalDeviceShaderObjectFeaturesEXT{}.setShaderObject(true).setPNext(vulkan11Features);
    auto swapchainMaintenanceFeatures = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT{}.setSwapchainMaintenance1(true).setPNext(shaderObjectFeatures);
    auto physicalDeviceFeatures2 = vk::PhysicalDeviceFeatures2{}.setPNext(swapchainMaintenanceFeatures);
    // * create device
    const auto device = std::make_shared<Device>(physicalDevice, dExtensions, Device::Queues{{queueFamilyIndex.value(), 1}}, &swapchainMaintenanceFeatures);

    // Vertex buffer setup
    const std::vector vertices = {
        -0.5f, -0.5f, 0.0f, 1.0f,
         0.5f, -0.5f, 0.0f, 1.0f,
         0.0f,  0.5f, 0.0f, 1.0f
    };
    const size_t verticesSize = vertices.size() * sizeof(float);
    const Buffer buffer{ device, verticesSize, {}, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible }; /* reBAR */
    void* const p = buffer.memory.mapMemory(0, vk::WholeSize);
    std::memcpy(p, vertices.data(), verticesSize);
    buffer.memory.unmapMemory();

    // Shader object setup : https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_shader_object.adoc
    constexpr vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint64_t) };
    const Shader shader{ device, { { vk::ShaderStageFlagBits::eVertex, shaders_spv, "vertexMain" }, { vk::ShaderStageFlagBits::eFragment, shaders_spv, "fragmentMain" } }, { pcRange } };

    // Swapchain setup
    Swapchain swapchain{ device, surface, queueFamilyIndex.value() };
    auto imgMemBarrier = vk::ImageMemoryBarrier2{}.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    const vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(imgMemBarrier);

    bool running = true, minimized = false;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) { running = false; }
            else if (event.type == SDL_EVENT_WINDOW_MINIMIZED) { minimized = true; }
            else if (event.type == SDL_EVENT_WINDOW_RESTORED) { swapchain.createSwapchain(); minimized = false; }
            else if (event.type == SDL_EVENT_WINDOW_RESIZED) { swapchain.createSwapchain(event.window.data1, event.window.data2); }
        }
        if (minimized) continue;
        
        swapchain.acquireNextImage();
        const auto& frame = swapchain.getCurrentFrame();
        const auto& cmdBuffer = frame.commandBuffer;

        imgMemBarrier.setImage(swapchain.getCurrentImage())
            .setOldLayout(vk::ImageLayout::eUndefined).setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands).setSrcAccessMask(vk::AccessFlagBits2::eNone)
            .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput).setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite);
        cmdBuffer.pipelineBarrier2(dependencyInfo);
        
        const auto rAttachmentInfo = vk::RenderingAttachmentInfo{ *swapchain.getCurrentImageView(), vk::ImageLayout::eColorAttachmentOptimal}
			.setLoadOp(vk::AttachmentLoadOp::eClear).setStoreOp(vk::AttachmentStoreOp::eStore).setClearValue(vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f });
        cmdBuffer.beginRendering({ {}, { {}, swapchain.extent() }, 1, 0, 1, &rAttachmentInfo });
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
            cmdBuffer.setViewportWithCountEXT({ { 0, 0, static_cast<float>(swapchain.extent().width), static_cast<float>(swapchain.extent().height) } });
            cmdBuffer.setScissorWithCountEXT({ { { 0, 0 }, swapchain.extent()}});
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

        imgMemBarrier.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal).setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput).setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eNone).setDstAccessMask(vk::AccessFlagBits2::eNone);
        cmdBuffer.pipelineBarrier2(dependencyInfo);
        swapchain.submitImage(device->queue[queueFamilyIndex.value()][0], vk::PipelineStageFlagBits::eColorAttachmentOutput);
    }
    device->waitIdle();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
