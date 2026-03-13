#pragma once

#include <fstream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

const int MAX_FRAMES_DRAWS = 3;

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// vertex data representation
struct Vertex {
  glm::vec3 pos; // position attribute for the vertex
  glm::vec3 col; // color attribute for the vertex
};

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
#

  // go back to the beginning of the file and read all the bytes at once
  file.seekg(0);
  file.read(fileBuffer.data(), fileSize);
  file.close();

  return fileBuffer;
}

static uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice, uint32_t allowedTypes,
    VkMemoryPropertyFlags
        properties) { // get properties of physical device memory
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    // check if memory type is among allowed types and has required properties
    if ((allowedTypes & (1 << i)) &&
        (memoryProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) { // desired properity bit flags are set
      return i;           // return index of memory type
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

static void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                         VkDeviceSize bufferSize,
                         VkBufferUsageFlags bufferUsage,
                         VkMemoryPropertyFlags bufferProperties,
                         VkBuffer *buffer, VkDeviceMemory *bufferMemory) {
  // information to create the buffer(assigining memory not included)
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;   // size of buffer in bytes
  bufferInfo.usage = bufferUsage; // type of buffer we want to create
  bufferInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // buffer is shared between multiple
                                 // queue families or not

  VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, buffer);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to create vertex buffer!");
  }

  // get memory requirements for the buffer to allocate enough memory
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);

  // allocate memory for the buffer
  VkMemoryAllocateInfo memoryAllocateInfo = {};
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.allocationSize =
      memRequirements.size; // size of allocation in bytes
  memoryAllocateInfo.memoryTypeIndex = findMemoryTypeIndex(
      physicalDevice,
      memRequirements.memoryTypeBits, // index of memory type on physical device
                                      // that has required bit flags
      bufferProperties);              // required memory property bit flags
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that we can map the memory and
  // write to it from the CPU and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means
  // that the memory will automatically be flushed

  // allocate memory for the buffer
  result = vkAllocateMemory(device, &memoryAllocateInfo, nullptr, bufferMemory);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate vertex buffer memory!");
  }

  // bind the buffer with the allocated memory
  vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
}

static void copyBuffer(VkDevice device, VkQueue transferQueue,
                       VkCommandPool transferCommandPool, VkBuffer srcBuffer,
                       VkBuffer dstBuffer, VkDeviceSize bufferSize) {
  // command buffer to hold transder commands
  VkCommandBuffer transferCommandBuffer;

  // command buffer allocation info
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = transferCommandPool;
  allocInfo.commandBufferCount = 1;

  // allocate command buffer from the command pool
  vkAllocateCommandBuffers(device, &allocInfo, &transferCommandBuffer);

  // info to begin the command buffer recording
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags =
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // only using it once so
                                                   // optimize for that

  // begin recording commands to the command buffer
  vkBeginCommandBuffer(transferCommandBuffer, &beginInfo);

  // region of data to copy and the buffers to copy between
  VkBufferCopy bufferCopyRegion = {};
  bufferCopyRegion.srcOffset = 0; // optional
  bufferCopyRegion.dstOffset = 0; // optional
  bufferCopyRegion.size = bufferSize;

  // command to copy src buffer to dst buffer and the region of data to copy
  vkCmdCopyBuffer(transferCommandBuffer, srcBuffer, dstBuffer, 1,
                  &bufferCopyRegion);

  vkEndCommandBuffer(transferCommandBuffer);

  // submit info about the command buffer to submit it to the queue
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &transferCommandBuffer;

  // submit transfer command buffer to the transfer queue and wait for it to finish
  vkQueueSubmit(transferQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(transferQueue); // wait for the transfer to finish before
                                  // cleaning up the command buffer

  // free the command buffer back to the command pool
  vkFreeCommandBuffers(device, transferCommandPool, 1, &transferCommandBuffer);
}
