#include "VulkanRenderer.h"
#include "Utilities.h"
#include "VulkanValidation.h"

#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <vulkan/vulkan_core.h>

VulkanRenderer::VulkanRenderer() {}

// helper functions for debug messenger
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// helper function to destroy the debug messenger
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
    func(instance, debugMessenger, pAllocator);
}

int VulkanRenderer::init(GLFWwindow *newWindow) {
  window = newWindow;

  // create vulkan instance
  try {
    createInstance();
    getPhysicalDevice();
    createLogicalDevice();
    createSurface();
  } catch (const std::runtime_error &e) {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

void VulkanRenderer::cleanup() {
  // Destroy the Logical Device first
  vkDestroyDevice(mainDevice.logicalDevice, nullptr);

  // Destroy the Surface
  vkDestroySurfaceKHR(instance, surface, nullptr);

  // Destroy the Debug Messenger
  if (enableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  }

  // destroy the Instance
  vkDestroyInstance(instance, nullptr);
}

VulkanRenderer::~VulkanRenderer() {}

void VulkanRenderer::createInstance() {
  // Check if the system/drivers actually have the validation layers
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("Validation layers requested, but not available!");
  }

  // info about the application
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan App";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  // create information for vkinstance
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // Get required extensions from GLFW
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> instanceExtensions(
      glfwExtensions, glfwExtensions + glfwExtensionCount);

  // Add the Debug Utils extension if validation is enabled
  //  This allows our custom callback to actually receive messages
  if (enableValidationLayers) {
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // Verify all requested extensions are supported by the GPU
  if (!checkInstanceExtensionSupport(&instanceExtensions)) {
    throw std::runtime_error("VkInstance does not support required extensions");
  }

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  // Hook the layers and the early debug messenger into creation
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // pNext part allows us to debug the actual creation and destruction
    // of the instance
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.pNext = nullptr;
  }

  // create the vulkan instance
  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  // 6. Setup the "permanent" debug messenger for the rest of the application
  if (enableValidationLayers) {
    if (CreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug messenger!");
    }
  }
}
void VulkanRenderer::createLogicalDevice() {
  // get the queue family indices for the physical device
  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);

  // queue the logical device needs to create and info to do so
  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex =
      indices
          .graphicsFamily; // index of the queue family to create a queue from
  queueCreateInfo.queueCount = 1; // number of queues to create
  float priority = 1.0f;
  queueCreateInfo.pQueuePriorities =
      &priority; // priority of the queues to create

  // info about the logical device to create
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1; // number of queue create infos
  deviceCreateInfo.pQueueCreateInfos =
      &queueCreateInfo; // list of queue create infos
  deviceCreateInfo.enabledExtensionCount =
      0; // number of device extensions to enable
  deviceCreateInfo.ppEnabledExtensionNames =
      nullptr; // list of enabled device extensions

  // physical device features
  VkPhysicalDeviceFeatures deviceFeatures = {};

  deviceCreateInfo.pEnabledFeatures =
      &deviceFeatures; // physical device features to use

  // create the logical device
  VkResult result = vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo,
                                   nullptr, &mainDevice.logicalDevice);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }

  // queues are created at the same time as the logical device
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0,
                   &graphicsQueue);
}

void VulkanRenderer::createSurface() {
  VkResult result =
      glfwCreateWindowSurface(instance, window, nullptr, &surface);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
}

void VulkanRenderer::getPhysicalDevice() {
  // enumerate physical devices the vkinstance can access
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("Failed to find a GPU with Vulkan support");
  }

  // get list of physical devices
  std::vector<VkPhysicalDevice> deviceList(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, deviceList.data());

  for (const auto &device : deviceList) {
    if (checkDeviceSuitable(device)) {
      mainDevice.physicalDevice = device;
      break;
    }
  }
}

bool VulkanRenderer::checkInstanceExtensionSupport(
    std::vector<const char *> *checkEtensions) {

  // get all available extensions
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  // create a list of all vkextension properties
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                         extensions.data());

  // check if all extensions in checkExtensions are in the list of available
  for (const auto &checkExtension : *checkEtensions) {
    bool hasExtension = false;

    for (const auto &extension : extensions) {
      if (strcmp(checkExtension, extension.extensionName) == 0) {
        hasExtension = true;
        break;
      }
    }
    if (!hasExtension) {
      return false;
    }
  }
  return true;
}

bool VulkanRenderer::checkDeviceSuitable(VkPhysicalDevice device) {
  /*
    // infor about the device
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures); */

  QueueFamilyIndices indices = getQueueFamilies(device);

  return indices.isValid();
}

QueueFamilyIndices VulkanRenderer::getQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  // get all the queue families of the device
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyList(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilyList.data());

  // go through each queue family and check if it has the required queues
  int i = 0;

  for (const auto &queueFamily : queueFamilyList) {
    // check if queue family has graphics capabilities
    if (queueFamily.queueCount > 0 &&
        queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i; // set graphics family index
    }

    // check if queue family is valid, if so break loop
    if (indices.isValid()) {
      break;
    }

    i++;
  }
  return indices;
}

bool VulkanRenderer::checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  // getting all the layers
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  // check for availability
  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound)
      return false;
  }

  return true;
}
