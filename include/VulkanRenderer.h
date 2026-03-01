#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

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
  VkInstance instance;

  VkDebugUtilsMessengerEXT debugMessenger;
  struct {
    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
  } mainDevice;
  VkQueue graphicsQueue;
  VkQueue presentationQueue;
  VkSurfaceKHR surface;

  // vulkan functions
  void createInstance();
  void createLogicalDevice();
  void createSurface();

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
};
