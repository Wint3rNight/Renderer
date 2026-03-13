#include "Mesh.h"
#include <cstring>
#include <vulkan/vulkan_core.h>

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
           std::vector<Vertex> *vertices) {
  vertexCount = vertices->size();
  physicalDevice = newPhysicalDevice;
  device = newDevice;
  createVertexBuffer(vertices);
}

int Mesh::getVertexCount() { return vertexCount; }

VkBuffer Mesh::getVertexBuffer() { return vertexBuffer; }

void Mesh::destroyVertexBuffer() {
  vkDestroyBuffer(device, vertexBuffer, nullptr);
  vkFreeMemory(device, vertexBufferMemory, nullptr);
}
Mesh::Mesh() {}
Mesh::~Mesh() {}

VkBuffer Mesh::createVertexBuffer(std::vector<Vertex> *vertices) {
  // information to create the buffer(assigining memory not included)
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size =
      sizeof(Vertex) * vertices->size(); // size of buffer in bytes
  bufferInfo.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; // type of buffer we want to create
  bufferInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // buffer is shared between multiple
                                 // queue families or not

  VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to create vertex buffer!");
  }

  // get memory requirements for the buffer to allocate enough memory
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

  // allocate memory for the buffer
  VkMemoryAllocateInfo memoryAllocateInfo = {};
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.allocationSize =
      memRequirements.size; // size of allocation in bytes
  memoryAllocateInfo.memoryTypeIndex = findMemoryTypeIndex(
      memRequirements.memoryTypeBits, // index of memory type on physical device
                                      // that has required bit flags
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that we can map the memory and
  // write to it from the CPU and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means
  // that the memory will automatically be flushed

  // allocate memory for the buffer
  result = vkAllocateMemory(device, &memoryAllocateInfo, nullptr,
                            &vertexBufferMemory);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate vertex buffer memory!");
  }

  // bind the buffer with the allocated memory
  vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

  // map memory to vertex buffer and copy vertex data to it
  void *data; // create a pointer to point in normal memory
  vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0,
              &data); // map the vertex buffer memory to the pointer
  memcpy(data, vertices->data(),
         (size_t)bufferInfo.size); // copy vertex data to mapped memory
  vkUnmapMemory(device, vertexBufferMemory); // unmap the memory

  return vertexBuffer;
}

uint32_t Mesh::findMemoryTypeIndex(
    uint32_t allowedTypes,
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
