#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "Utilities.h"

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void cleanup();

  ~VulkanRenderer();

private:
  GLFWwindow *window;

  // vk components
  // main
  VkInstance instance;

  VkDebugUtilsMessengerEXT debugMessenger;
  struct {
    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
  } mainDevice;
  VkQueue graphicsQueue;
  VkQueue presentationQueue;
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;
  std::vector<SwapChainImage> swapChainImages;

  // utils
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  // vulkan functions
  // create functions
  void createInstance();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();

  // get functions
  void getPhysicalDevice();

  // support functions
  // checker functions
  bool checkInstanceExtensionSupport(std::vector<const char *> *checkEtensions);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  bool checkValidationLayerSupport();
  bool checkDeviceSuitable(VkPhysicalDevice device);

  // getter functions
  QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device);
  SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

  // choose functions
  VkSurfaceFormatKHR
  chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
  VkPresentModeKHR chooseBestPresentationMode(
      const std::vector<VkPresentModeKHR> &presentationModes);
  VkExtent2D
  chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities);

  // support create functions
  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);
};
