#include "Mesh.h"
#include "Utilities.h"
#include <cstring>
#include <vulkan/vulkan_core.h>

Mesh::Mesh() {}

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
           VkQueue transferQueue, VkCommandPool transferCommandPool,
           std::vector<Vertex> *vertices, std::vector<uint32_t> *indices) {
  vertexCount = vertices->size();
  indexCount = indices->size();
  physicalDevice = newPhysicalDevice;
  device = newDevice;
  createVertexBuffer(transferQueue, transferCommandPool, vertices);
  createIndexBuffer(transferQueue, transferCommandPool, indices);

  uboModel.model = glm::mat4(1.0f);
}

void Mesh::setModel(glm::mat4 newModel) { uboModel.model = newModel; }

UboModel Mesh::getModel() { return uboModel; }

int Mesh::getVertexCount() { return vertexCount; }
VkBuffer Mesh::getVertexBuffer() { return vertexBuffer; }
int Mesh::getIndexCount() { return indexCount; }
VkBuffer Mesh::getIndexBuffer() { return indexBuffer; }

void Mesh::destroyBuffers() {
  vkDestroyBuffer(device, vertexBuffer, nullptr);
  vkFreeMemory(device, vertexBufferMemory, nullptr);
  vkDestroyBuffer(device, indexBuffer, nullptr);
  vkFreeMemory(device, indexBufferMemory, nullptr);
}
Mesh::~Mesh() {}

void Mesh::createVertexBuffer(VkQueue transferQueue,
                              VkCommandPool transferCommandPool,
                              std::vector<Vertex> *vertices) {

  // get size of buffer in bytes
  VkDeviceSize bufferSize = sizeof(Vertex) * vertices->size();

  // temp buffer to stage vertex data before transfering to the vertex buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  // create the vertex buffer and allocate memory for it
  createBuffer(physicalDevice, device, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stagingBuffer, &stagingBufferMemory);

  // map memory to vertex buffer and copy vertex data to it
  void *data; // create a pointer to point in normal memory
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0,
              &data); // map the vertex buffer memory to the pointer
  memcpy(data, vertices->data(),
         (size_t)bufferSize); // copy vertex data to mapped memory
  vkUnmapMemory(device, stagingBufferMemory); // unmap the memory

  // create buffer with transfer_destination_bit to mark as recepient of
  // transfer(also vertex buffer) and device_local_bit since buffer will be used
  // on the GPU and not updated very often. Only accessible from the GPU, not
  // the CPU
  createBuffer(
      physicalDevice, device, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, &vertexBufferMemory);

  // copy the vertex buffer data from the staging buffer to the vertex buffer
  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer,
             vertexBuffer, bufferSize);

  // clean up staging buffer and its memory since data has been transfered
  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Mesh::createIndexBuffer(VkQueue transferQueue,
                             VkCommandPool transferCommandPool,
                             std::vector<uint32_t> *indices) {
  // get size of buffer in bytes
  VkDeviceSize bufferSize = sizeof(uint32_t) * indices->size();

  // temp buffer to stage index data before transfering to the index buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  // create the index buffer and allocate memory for it
  createBuffer(physicalDevice, device, bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stagingBuffer, &stagingBufferMemory);

  // map memory to index buffer and copy index data to it
  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices->data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  // create buffer for index data on GPu access only area
  createBuffer(
      physicalDevice, device, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, &indexBufferMemory);

  // copy the index buffer data from the staging buffer to the vertex buffer
  copyBuffer(device, transferQueue, transferCommandPool, stagingBuffer,
             indexBuffer, bufferSize);

  // clean up staging buffer and its memory since data has been transfered
  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}
