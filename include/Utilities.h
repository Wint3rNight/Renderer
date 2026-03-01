#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <vulkan/vulkan_core.h>

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Keep only the general utility structures here
struct QueueFamilyIndices {
  int graphicsFamily = -1;     // location of the graphics queue family
  int presentationFamily = -1; // location of the presentation queue family

  // check if the queue family indices are valid
  bool isValid() { return graphicsFamily >= 0 && presentationFamily >= 0; }
};

struct SwapChainDetails{
  VkSurfaceCapabilitiesKHR surfaceCapabilities; //surafece properties like image size/extent
  std::vector<VkSurfaceFormatKHR> formats; // surface image formats, color depth, etc
  std::vector<VkPresentModeKHR> presentModes; // how images should be presented to the screen
};
