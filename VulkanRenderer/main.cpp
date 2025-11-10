#include <iostream>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#undef max

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation", 
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
       // VK_KHR_SPIRV_1_4_EXTENSION_NAME,//vk::KHRSpirv14ExtensionName,,
        //VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,//vk::KHRSynchronization2ExtensionName,
        //VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,//vk::KHRCreateRenderpass2ExtensionName
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = true;
#else
constexpr bool enableValidationLayers = true;
#endif

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context context{};
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    uint32_t graphicsFamily = 0, presentFamily = 0;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::SurfaceFormatKHR swapChainSurfaceFormat{};
    vk::Extent2D swapChainExtent{};
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages{};
    std::vector<vk::raii::ImageView> swapChainImageViews{};

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Vulkan Renderer";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = vk::ApiVersion;

        // Get the required layers
        std::vector<char const*> requiredLayers{};
        if (enableValidationLayers) {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::any_of(requiredLayers.begin(), requiredLayers.end(),
            [&layerProperties](auto const& requiredLayer)
            {
                return std::none_of(layerProperties.begin(), layerProperties.end(),
                    [requiredLayer](auto const& layerProperty)
                    {
                        return strcmp(layerProperty.layerName, requiredLayer) == 0;
                    });
            }))
        {
            throw std::runtime_error("One or more required layers are not supported!");
        }

        // Get the required instance extensions from GLFW.
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Check if the required GLFW extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (uint32_t i = 0; i < glfwExtensionCount; ++i)
        {
            auto glfwExtension = glfwExtensions[i];
            if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
                [glfwExtension](auto const& extensionProperty)
                {
                    return std::strcmp(extensionProperty.extensionName, glfwExtension) == 0;
                }))
            {
                throw std::runtime_error(std::string("Required GLFW extension not supported: ") + glfwExtension);
            }
        }

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
        createInfo.ppEnabledLayerNames = requiredLayers.data();
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        instance = vk::raii::Instance(context, createInfo);
    }

    void pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::find_if(devices.begin(), devices.end(),
            [&](auto const& device)
            {
                auto queueFamilies = device.getQueueFamilyProperties();
                auto isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
                const auto qfpIter = std::find_if(queueFamilies.begin(), queueFamilies.end(),
                    [](vk::QueueFamilyProperties const& qfp)
                    {
                        return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                    });
                isSuitable = isSuitable && (qfpIter != queueFamilies.end());
                auto extensions = device.enumerateDeviceExtensionProperties();
                auto found = true;
                for (auto const& extension : deviceExtensions) {
                    auto extensionIter = std::find_if(extensions.begin(), extensions.end(),
                        [extension](auto const& ext)
                        {
                            return strcmp(ext.extensionName, extension) == 0;
                        });
                    found = found && extensionIter != extensions.end();
                }
                isSuitable = isSuitable && found;
                if (isSuitable) {
                    physicalDevice = device;
                }
                return isSuitable;
            });
        if (devIter == devices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        std::tie(graphicsFamily, presentFamily) = findQueueFamilies(*physicalDevice);
        auto queuePriority = 0.0f;

        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
        deviceQueueCreateInfo.queueFamilyIndex = graphicsFamily;
        deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

        vk::PhysicalDeviceFeatures2 features2{}; // vk::PhysicalDeviceFeatures2 (empty for now)
        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.dynamicRendering = true; // Enable dynamic rendering from Vulkan 1.3
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynamicStateFeatures{};
        extDynamicStateFeatures.extendedDynamicState = true; // Enable extended dynamic state from the extension

        // Create a chain of feature structures
        auto featureChain = vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        { features2, vulkan13Features, extDynamicStateFeatures };

        vk::DeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsFamily, 0);
        presentQueue = vk::raii::Queue(device, presentFamily, 0);
    }

    std::tuple<uint32_t, uint32_t> findQueueFamilies(VkPhysicalDevice device) {
        // find the index of the first queue family that supports graphics
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
            [](vk::QueueFamilyProperties const& qfp)
            {
                return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
            });

        auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface)
            ? graphicsIndex
            : static_cast<uint32_t>(queueFamilyProperties.size());

        if (presentIndex == queueFamilyProperties.size())
        {
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for (size_t i = 0; i < queueFamilyProperties.size(); i++)
            {
                if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                    physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
                {
                    graphicsIndex = static_cast<uint32_t>(i);
                    presentIndex = graphicsIndex;
                    break;
                }
            }
            if (presentIndex == queueFamilyProperties.size())
            {
                // there's nothing like a single family index that supports both graphics and present -> look for another
                // family index that supports present
                for (size_t i = 0; i < queueFamilyProperties.size(); i++)
                {
                    if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
                    {
                        presentIndex = static_cast<uint32_t>(i);
                        break;
                    }
                }
            }
        }

        if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size()))
        {
            throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
        }

        return { graphicsIndex, presentIndex };
    }

    void createSurface() {
        VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = glfwGetWin32Window(window);
        createInfo.hinstance = GetModuleHandle(nullptr);

        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
        auto availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);

        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);

        vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
        swapChainCreateInfo.flags = vk::SwapchainCreateFlagsKHR();
        swapChainCreateInfo.surface = surface;
        swapChainCreateInfo.minImageCount = chooseSwapMinImageCount(surfaceCapabilities);
        swapChainCreateInfo.imageFormat = swapChainSurfaceFormat.format;
        swapChainCreateInfo.imageColorSpace = swapChainSurfaceFormat.colorSpace;
        swapChainCreateInfo.imageExtent = swapChainExtent;
        swapChainCreateInfo.imageArrayLayers = 1; // keep 1 unless rendering for VR
        swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // we are rendering to image directly
        swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;  // don't apply further transformation
        swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque; // don�t blend with other windows in the system
        swapChainCreateInfo.presentMode = chooseSwapPresentMode(availablePresentModes);
        swapChainCreateInfo.clipped = true;  // don�t update the pixels that are obscured

        uint32_t queueFamilyIndices[] = { graphicsFamily, presentFamily };

        if (graphicsFamily != presentFamily) {
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            swapChainCreateInfo.queueFamilyIndexCount = 2;
            swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
        auto minImageCount = surfaceCapabilities.minImageCount + 1;
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imageViewCreateInfo.format = swapChainSurfaceFormat.format;
        imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        imageViewCreateInfo.components.r = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.g = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.b = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.a = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        for (auto image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}