#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include "Material.h"
#include "Utilities.h"

class VulkanDevice;
class DescriptorManager;

class TextureManager {
public:
  TextureManager() = default;
  ~TextureManager() = default;

  TextureManager(const TextureManager &) = delete;
  TextureManager &operator=(const TextureManager &) = delete;

  void init(const VulkanDevice &device, VkPipelineCache pipelineCache);
  void cleanup(VkDevice logicalDevice, VmaAllocator allocator);

  int loadTexture(const std::string &filename, const VulkanDevice &device,
                  DescriptorManager &descriptorManager);

  Material loadMaterial(const std::string &albedoFilename,
                        const std::string &normalFilename,
                        const std::string &metallicFilename,
                        const std::string &roughnessFilename,
                        const std::string &aoFilename,
                        const VulkanDevice &device,
                        DescriptorManager &descriptorManager);

  int loadSkybox(const std::string &filename, const VulkanDevice &device,
                 DescriptorManager &descriptorManager);

  void setModelDirectory(const std::string &dir) { modelBaseDirectory = dir; }

  // IBL: generates irradiance + prefiltered env maps from the loaded skybox cubemap.
  // Must be called after loadSkybox(). Returns the raw image index in textureImages[].
  int createIrradianceMap(int skyboxImageIndex, const VulkanDevice &device);
  int createPrefilteredEnvMap(int skyboxImageIndex, const VulkanDevice &device);
  int loadBrdfLut(const VulkanDevice &device);
  int createSsaoNoiseTexture(const VulkanDevice &device);

  VkImageView getImageView(int index) const { return textureImageViews[index].get(); }
  VkSampler   getTextureSampler() const { return textureSampler; }
  VkSampler   getSkyboxSampler()  const { return skyboxSampler; }

private:
  VkSampler textureSampler = VK_NULL_HANDLE;
  VkSampler skyboxSampler  = VK_NULL_HANDLE;

  std::vector<AllocatedImage>  textureImages;
  std::vector<ImageViewHandle> textureImageViews;

  std::unordered_map<std::string, int> textureMap;
  std::unordered_map<std::string, int> materialMap;
  std::unordered_map<std::string, int> skyboxMap;

  void createTextureSampler(VkDevice logicalDevice, VkPhysicalDevice physicalDevice);
  void createSkyboxSampler(VkDevice logicalDevice, VkPhysicalDevice physicalDevice);

  int  getOrCreateTexture(const std::string &filename, const VulkanDevice &device);
  int  getOrCreateFlatTexture(const std::string &cacheKey, uint8_t r, uint8_t g, uint8_t b, uint8_t a, const VulkanDevice &device);
  int  createTextureImage(const std::string &filename,
                          const std::string &cacheKey,
                          const VulkanDevice &device);
  // shareWithCompute: CONCURRENT graphics+compute sharing for textures the
  // dedicated compute queue samples (e.g. SSAO noise).
  int  createTextureImageFromPixels(const std::string &cacheKey,
                                    const unsigned char *pixels, int width,
                                    int height, const VulkanDevice &device,
                                    bool shareWithCompute = false);
  unsigned char *loadTextureFile(const std::string &filename, int *width,
                                 int *height, VkDeviceSize *imageSize);
  std::string resolveTexturePath(const std::string &filename) const;
  std::string normalizeTextureKey(const std::string &filename) const;
  int createHdrCubemap(const std::string &filename, const VulkanDevice &device,
                       uint32_t faceSize);

  std::string modelBaseDirectory;  // set before loading each model's materials

  // GPU compute IBL precomputation
  VkPipeline            irradiancePipeline = VK_NULL_HANDLE;
  VkPipeline            prefilterPipeline  = VK_NULL_HANDLE;
  VkPipelineLayout      computeLayout      = VK_NULL_HANDLE;
  VkDescriptorSetLayout computeSetLayout   = VK_NULL_HANDLE;
  VkDescriptorPool      computePool        = VK_NULL_HANDLE;

  void createComputePipelines(const VulkanDevice &device,
                              VkPipelineCache pipelineCache);
  void destroyComputePipelines(VkDevice device);
  int  dispatchIrradianceCompute(int srcImageIndex, const VulkanDevice &device);
  int  dispatchPrefilterCompute(int srcImageIndex, const VulkanDevice &device);
};
