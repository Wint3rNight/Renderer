#pragma once

#include <fstream>

#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

const int MAX_FRAMES_DRAWS = 2;

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// Keep only the general utility structures here
struct QueueFamilyIndices {
  int graphicsFamily = -1;     // location of the graphics queue family
  int presentationFamily = -1; // location of the presentation queue family

  // check if the queue family indices are valid
  bool isValid() { return graphicsFamily >= 0 && presentationFamily >= 0; }
};

struct SwapChainDetails {
  VkSurfaceCapabilitiesKHR
      surfaceCapabilities; // surafece properties like image size/extent
  std::vector<VkSurfaceFormatKHR>
      formats; // surface image formats, color depth, etc
  std::vector<VkPresentModeKHR>
      presentModes; // how images should be presented to the screen
};

struct SwapChainImage {
  VkImage image;
  VkImageView imageView;
};

static std::vector<char> readFile(const std::string &filename) {
  // open file at the end to get the size, and in binary mode
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // get file size by reading the position of the cursor
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> fileBuffer(fileSize);

  // go back to the beginning of the file and read all the bytes at once
  file.seekg(0);
  file.read(fileBuffer.data(), fileSize);
  file.close();

  return fileBuffer;
}
