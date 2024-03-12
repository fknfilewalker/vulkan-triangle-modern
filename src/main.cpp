#include <deque>
#include <vulkan/vulkan_raii.hpp>
#ifdef __APPLE__
constexpr bool isApple = true;
#else
constexpr bool isApple = false;
#endif
#include <GLFW/glfw3.h>

#include <optional>
#include <algorithm>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <memory>

constexpr bool fullscreen = false;
constexpr uint32_t window_width = 800;
constexpr uint32_t window_height = 600;

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

struct Device
{
    using QueueFamily = uint16_t;
    using QueueCount = uint16_t;
    using Queues = std::unordered_map<QueueFamily, QueueCount>;
    Device(const vk::raii::PhysicalDevice& physicalDevice, const std::vector<const char*>& extensions, const Queues& queues, const void* pNext) :
		device{ nullptr }, physicalDevice{ physicalDevice }, memoryProperties{ physicalDevice.getMemoryProperties() }
    {
        constexpr float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
        deviceQueueCreateInfos.reserve(queues.size());
        for (const auto& [queueFamilyIndex, queueCount] : queues) {
            deviceQueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queueCount, &priority });
        }
        const vk::DeviceCreateInfo deviceCreateInfo{ {}, deviceQueueCreateInfos, {}, extensions,{}, pNext };
        device = vk::raii::Device{ physicalDevice, deviceCreateInfo };
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
            if ((requirements.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
        return std::nullopt;
    }

    operator const vk::raii::Device& () const { return device; }
    operator const vk::raii::PhysicalDevice& () const { return physicalDevice; }

    vk::raii::Device device;
    std::vector<std::vector<vk::raii::Queue>> queue;
    vk::raii::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceMemoryProperties memoryProperties;
};

// Every resource has a device reference
struct Resource { std::shared_ptr<Device> dev; };



// Data for one frame/image in our swapchain
struct Frame {
    Frame(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool) :
        acquireFence{ device, vk::FenceCreateInfo{} },
        submitFence{ device, vk::FenceCreateInfo{} },
    	presentFinishFence{ device, vk::FenceCreateInfo{} },
    	imageAvailableSemaphore{ device, vk::SemaphoreCreateInfo{} },
    	renderFinishedSemaphore{ device, vk::SemaphoreCreateInfo{} },
    	commandBuffer{ std::move(vk::raii::CommandBuffers{ device, { *commandPool, vk::CommandBufferLevel::ePrimary, 1 } }[0]) }
    {}
    vk::raii::Fence acquireFence, submitFence, presentFinishFence;
    vk::raii::Semaphore imageAvailableSemaphore, renderFinishedSemaphore;
    vk::raii::CommandBuffer commandBuffer;
};


int main(int /*argc*/, char** /*argv*/)
{
    if (!glfwInit()) exitWithError("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWmonitor* monitor = fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Vulkan Triangle Modern", monitor, nullptr);

    const vk::raii::Context context{};
    constexpr vk::ApplicationInfo applicationInfo{ nullptr, 0, nullptr, 0, vk::ApiVersion13 };

    // Instance Setup
    std::vector<const char*> iExtensions{ vk::EXTSurfaceMaintenance1ExtensionName, vk::KHRGetSurfaceCapabilities2ExtensionName };
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
    vk::InstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(iExtensions);
    instanceCreateInfo.setPEnabledLayerNames(iLayers);
    if constexpr (isApple) instanceCreateInfo.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);

    const vk::raii::Instance instance(context, instanceCreateInfo);

    // Surface Setup
    VkSurfaceKHR _surface;
    glfwCreateWindowSurface(*instance, window, nullptr, &_surface);
    vk::raii::SurfaceKHR surfaceKHR = vk::raii::SurfaceKHR{ instance, _surface };
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
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{ true };
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance{ true, &bufferDeviceAddressFeatures };
    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ true, &swapchainMaintenance };
    vk::PhysicalDeviceSynchronization2Features synchronization2Features{ true, &shaderObjectFeatures };
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{ true, &synchronization2Features };
    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2{ {}, &dynamicRenderingFeatures };
    physicalDeviceFeatures2.features.shaderInt64 = true;
    // * create device
    auto device = std::make_shared<Device>(physicalDevice, dExtensions, Device::Queues{ {queueFamilyIndex.value(), 1} }, &physicalDeviceFeatures2);

    vk::ImageMemoryBarrier2 imageMemoryBarrier{};
    imageMemoryBarrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(imageMemoryBarrier);

    // Swapchain setup
    const auto surfaceCapabilities = device->physicalDevice.getSurfaceCapabilitiesKHR(*surfaceKHR);
    const auto surfaceFormats = device->physicalDevice.getSurfaceFormatsKHR(*surfaceKHR);
    auto imageCount = std::min(3u, surfaceCapabilities.maxImageCount);
    auto extent = surfaceCapabilities.currentExtent;
    vk::SwapchainCreateInfoKHR swapchainCreateInfoKHR{ {}, *surfaceKHR, imageCount,
        surfaceFormats[0].format, surfaceFormats[0].colorSpace, extent,
        1u, vk::ImageUsageFlagBits::eColorAttachment };
    //swapchainCreateInfoKHR.setPresentMode(vk::PresentModeKHR::eImmediate);
    auto swapchainKHR = vk::raii::SwapchainKHR{ *device, swapchainCreateInfoKHR };
    auto images = swapchainKHR.getImages();
    vk::raii::CommandPool commandPool{ *device, { vk::CommandPoolCreateFlagBits::eTransient, queueFamilyIndex.value() } };

    { // frame
        glfwPollEvents();

        Frame frame{ *device, commandPool };

        auto [result, image_idx] = swapchainKHR.acquireNextImage(UINT64_MAX, *frame.imageAvailableSemaphore);
        resultCheck(result, "OH NO");

        frame.commandBuffer.begin({ });
        imageMemoryBarrier.image = images[image_idx];
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
        imageMemoryBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        frame.commandBuffer.pipelineBarrier2(dependencyInfo);
        frame.commandBuffer.end();

        constexpr vk::PipelineStageFlags waitDstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        device->queue[queueFamilyIndex.value()][0].submit(vk::SubmitInfo{ *frame.imageAvailableSemaphore, waitDstStageMask,
                *frame.commandBuffer,*frame.renderFinishedSemaphore });// , * frame.submitFence);

        vk::SwapchainPresentFenceInfoEXT presentFenceInfo{ *frame.presentFinishFence };
        resultCheck(device->queue[queueFamilyIndex.value()][0].presentKHR({ *frame.renderFinishedSemaphore, *swapchainKHR, image_idx, {}, &presentFenceInfo }), "present swapchain image error");
        auto status = frame.presentFinishFence.getStatus();
        while(status != vk::Result::eSuccess) status = frame.presentFinishFence.getStatus();
        // delete all frame resources
        frame = Frame{ *device, commandPool };
    }
    
    device->device.waitIdle();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
