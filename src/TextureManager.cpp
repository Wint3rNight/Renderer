#include "TextureManager.h"
#include "DescriptorManager.h"
#include "RenderResources.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanSync.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr float ALPHA_TEST_THRESHOLD = 0.5f; // glTF MASK default cutoff

// 2x2 box downsample of RGBA8 to half size. Source can be non-square; odd
// dimensions clamp to edge.
std::vector<uint8_t> boxDownsampleRGBA8(const std::vector<uint8_t> &src,
                                        int sw, int sh, int &dw, int &dh) {
  dw = std::max(1, sw / 2);
  dh = std::max(1, sh / 2);
  std::vector<uint8_t> dst(static_cast<size_t>(dw) * dh * 4);
  for (int y = 0; y < dh; ++y) {
    int sy0 = std::min(y * 2,     sh - 1);
    int sy1 = std::min(y * 2 + 1, sh - 1);
    for (int x = 0; x < dw; ++x) {
      int sx0 = std::min(x * 2,     sw - 1);
      int sx1 = std::min(x * 2 + 1, sw - 1);
      const uint8_t *p00 = &src[(static_cast<size_t>(sy0) * sw + sx0) * 4];
      const uint8_t *p10 = &src[(static_cast<size_t>(sy0) * sw + sx1) * 4];
      const uint8_t *p01 = &src[(static_cast<size_t>(sy1) * sw + sx0) * 4];
      const uint8_t *p11 = &src[(static_cast<size_t>(sy1) * sw + sx1) * 4];
      uint8_t *d = &dst[(static_cast<size_t>(y) * dw + x) * 4];
      for (int c = 0; c < 4; ++c)
        d[c] = static_cast<uint8_t>(
            (static_cast<int>(p00[c]) + p10[c] + p01[c] + p11[c] + 2) / 4);
    }
  }
  return dst;
}

// Fraction of pixels with (alpha * scale) >= threshold.
float computeAlphaCoverage(const std::vector<uint8_t> &px, float scale,
                           float threshold) {
  const size_t n = px.size() / 4;
  if (n == 0) return 0.0f;
  size_t covered = 0;
  const float t = threshold * 255.0f;
  for (size_t i = 0; i < n; ++i)
    if (static_cast<float>(px[i * 4 + 3]) * scale >= t) ++covered;
  return static_cast<float>(covered) / static_cast<float>(n);
}

// Castano 2010 — binary-search an alpha scale that makes this mip's alpha
// coverage match mip 0's at the engine's alpha-test threshold. Without this,
// linearly filtered alpha shrinks toward the cutout center at higher mips
// and foliage silhouettes shimmer at distance.
float findAlphaScaleForCoverage(const std::vector<uint8_t> &px,
                                float targetCoverage, float threshold) {
  float lo = 0.0f, hi = 4.0f;
  // 12 iterations gives ~1/4096 precision on the scale — well past JND.
  for (int i = 0; i < 12; ++i) {
    float mid = 0.5f * (lo + hi);
    float cov = computeAlphaCoverage(px, mid, threshold);
    if (cov > targetCoverage) hi = mid;
    else lo = mid;
  }
  return 0.5f * (lo + hi);
}

void rescaleAlpha(std::vector<uint8_t> &px, float scale) {
  const size_t n = px.size() / 4;
  for (size_t i = 0; i < n; ++i) {
    float a = static_cast<float>(px[i * 4 + 3]) * scale;
    if (a < 0.0f) a = 0.0f;
    if (a > 255.0f) a = 255.0f;
    px[i * 4 + 3] = static_cast<uint8_t>(a + 0.5f);
  }
}

glm::vec3 faceDirection(uint32_t face, float u, float v) {
  switch (face) {
  case 0:
    return glm::normalize(glm::vec3(1.0f, -v, -u));
  case 1:
    return glm::normalize(glm::vec3(-1.0f, -v, u));
  case 2:
    return glm::normalize(glm::vec3(u, 1.0f, v));
  case 3:
    return glm::normalize(glm::vec3(u, -1.0f, -v));
  case 4:
    return glm::normalize(glm::vec3(u, -v, 1.0f));
  default:
    return glm::normalize(glm::vec3(-u, -v, -1.0f));
  }
}

glm::vec4 sampleEquirectangular(const float *pixels, int width, int height,
                                const glm::vec3 &direction) {
  float phi = std::atan2(direction.z, direction.x);
  float theta = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
  float uf = (phi + PI) / (2.0f * PI) * static_cast<float>(width - 1);
  float vf = (0.5f - theta / PI) * static_cast<float>(height - 1);

  int x0 = static_cast<int>(std::floor(uf));
  int y0 = static_cast<int>(std::floor(vf));
  int x1 = (x0 + 1) % width;
  int y1 = std::min(y0 + 1, height - 1);
  float tx = uf - static_cast<float>(x0);
  float ty = vf - static_cast<float>(y0);

  auto texel = [&](int x, int y) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 4;
    return glm::vec4(pixels[idx], pixels[idx + 1], pixels[idx + 2],
                     pixels[idx + 3]);
  };
  glm::vec4 a = glm::mix(texel(x0, y0), texel(x1, y0), tx);
  glm::vec4 b = glm::mix(texel(x0, y1), texel(x1, y1), tx);
  return glm::mix(a, b, ty);
}

// Layout transition helper for layered/mipped images
static void transitionCubemapLayout(VkDevice device, VkQueue queue,
                                    VkCommandPool pool, VkImage image,
                                    VkImageLayout oldLayout,
                                    VkImageLayout newLayout,
                                    uint32_t layerCount,
                                    uint32_t levelCount = 1) {
  VkCommandBuffer cmd = beginCommandBuffer(device, pool);

  VkImageMemoryBarrier b = {};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.oldLayout = oldLayout;
  b.newLayout = newLayout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levelCount, 0,
                        layerCount};

  VkPipelineStageFlags srcStage, dstStage;

  auto srcFor = [&](VkImageLayout l) -> VkAccessFlags {
    switch (l) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return 0;
    case VK_IMAGE_LAYOUT_GENERAL:
      return VK_ACCESS_SHADER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
    default:
      return VK_ACCESS_MEMORY_WRITE_BIT;
    }
  };
  auto dstFor = [&](VkImageLayout l) -> VkAccessFlags {
    switch (l) {
    case VK_IMAGE_LAYOUT_GENERAL:
      return VK_ACCESS_SHADER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
    default:
      return VK_ACCESS_MEMORY_READ_BIT;
    }
  };
  auto stageFor = [&](VkImageLayout l, bool /*isSrc*/) -> VkPipelineStageFlags {
    switch (l) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    case VK_IMAGE_LAYOUT_GENERAL:
      return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    default:
      return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
  };

  b.srcAccessMask = srcFor(oldLayout);
  b.dstAccessMask = dstFor(newLayout);
  srcStage = stageFor(oldLayout, true);
  dstStage = stageFor(newLayout, false);

  recordImageBarrier2(cmd, srcStage, dstStage, b);
  endAndSubmitCommandBuffer(device, pool, queue, cmd);
}

static VkShaderModule loadSpv(VkDevice device, const std::string &relPath) {
  std::vector<std::string> candidates = {relPath, "../" + relPath};
  std::string found;
  for (const auto &c : candidates)
    if (std::filesystem::exists(c)) {
      found = c;
      break;
    }
  if (found.empty())
    throw std::runtime_error("Compute shader not found: " + relPath +
                             ". Run compile_shaders.sh");

  std::ifstream file(found, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open shader: " + found);
  size_t sz = static_cast<size_t>(file.tellg());
  std::vector<char> code(sz);
  file.seekg(0);
  file.read(code.data(), static_cast<std::streamsize>(sz));

  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = sz;
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule mod;
  if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module: " + found);
  return mod;
}
} // namespace

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

void TextureManager::init(const VulkanDevice &device,
                          VkPipelineCache pipelineCache) {
  createTextureSampler(device.getLogicalDevice(), device.getPhysicalDevice());
  createSkyboxSampler(device.getLogicalDevice(), device.getPhysicalDevice());
  createComputePipelines(device, pipelineCache);
}

void TextureManager::cleanup(VkDevice logicalDevice,
                             VmaAllocator /*allocator*/) {
  destroyComputePipelines(logicalDevice);
  textureImageViews.clear();
  textureImages.clear();
  textureMap.clear();
  materialMap.clear();
  skyboxMap.clear();
  if (skyboxSampler != VK_NULL_HANDLE) {
    vkDestroySampler(logicalDevice, skyboxSampler, nullptr);
    skyboxSampler = VK_NULL_HANDLE;
  }
  if (textureSampler != VK_NULL_HANDLE) {
    vkDestroySampler(logicalDevice, textureSampler, nullptr);
    textureSampler = VK_NULL_HANDLE;
  }
}

int TextureManager::loadTexture(const std::string &filename,
                                const VulkanDevice &device,
                                DescriptorManager &descriptorManager) {
  return loadMaterial(filename, "", "", "", "", device, descriptorManager)
      .descriptorSetId;
}

Material TextureManager::loadMaterial(const std::string &albedoFilename,
                                      const std::string &normalFilename,
                                      const std::string &metallicFilename,
                                      const std::string &roughnessFilename,
                                      const std::string &aoFilename,
                                      const VulkanDevice &device,
                                      DescriptorManager &descriptorManager) {
  std::string albedoKey = albedoFilename.empty()
                              ? "plain.png"
                              : normalizeTextureKey(albedoFilename);
  std::string normalKey = normalFilename.empty()
                              ? "__flat_normal__"
                              : normalizeTextureKey(normalFilename);
  std::string metallicKey = metallicFilename.empty()
                                ? "__flat_black__"
                                : normalizeTextureKey(metallicFilename);
  std::string roughnessKey = roughnessFilename.empty()
                                 ? "__flat_white__"
                                 : normalizeTextureKey(roughnessFilename);
  std::string aoKey = aoFilename.empty() ? "__flat_white_ao__"
                                         : normalizeTextureKey(aoFilename);

  // Directory-qualified like the texture cache: equal bare names from two
  // different models must not alias each other's material descriptors.
  std::string materialKey = modelBaseDirectory + "|" + albedoKey + "|" +
                            normalKey + "|" + metallicKey + "|" +
                            roughnessKey + "|" + aoKey;

  auto resolveTex = [&](const std::string &key, uint8_t r, uint8_t g, uint8_t b,
                        uint8_t a) {
    if (key == "__flat_normal__" || key == "__flat_black__" ||
        key == "__flat_white__" || key == "__flat_white_ao__")
      return getOrCreateFlatTexture(key, r, g, b, a, device);
    return getOrCreateTexture(key, device);
  };

  auto matIt = materialMap.find(materialKey);
  if (matIt != materialMap.end()) {
    Material cached;
    cached.descriptorSetId = matIt->second;
    cached.albedoTextureId = resolveTex(albedoKey, 255, 255, 255, 255);
    cached.normalTextureId = resolveTex(normalKey, 128, 128, 255, 255);
    cached.metallicTextureId = resolveTex(metallicKey, 0, 0, 0, 255);
    cached.roughnessTextureId = resolveTex(roughnessKey, 255, 255, 255, 255);
    cached.aoTextureId = resolveTex(aoKey, 255, 255, 255, 255);
    return cached;
  }

  Material mat;
  mat.albedoTextureId = resolveTex(albedoKey, 255, 255, 255, 255);
  mat.normalTextureId = resolveTex(normalKey, 128, 128, 255, 255);
  mat.metallicTextureId = resolveTex(metallicKey, 0, 0, 0, 255);
  mat.roughnessTextureId = resolveTex(roughnessKey, 255, 255, 255, 255);
  mat.aoTextureId = resolveTex(aoKey, 255, 255, 255, 255);

  // Phase 7.2: register each texture into the global bindless array.
  // The bindless index == the raw texture image index in textureImages[].
  // This is called once per unique material; duplicate registrations
  // (same texture appearing in multiple materials) are idempotent writes
  // to the same descriptor slot.
  auto regBindless = [&](int texId) {
    descriptorManager.registerBindlessTexture(
        device.getLogicalDevice(), static_cast<uint32_t>(texId),
        textureImageViews[texId].get(), textureSampler);
  };
  regBindless(mat.albedoTextureId);
  regBindless(mat.normalTextureId);
  regBindless(mat.metallicTextureId);
  regBindless(mat.roughnessTextureId);
  regBindless(mat.aoTextureId);

  // Legacy: still allocate a per-material descriptor set for any code paths
  // that haven't been converted yet.
  mat.descriptorSetId = descriptorManager.createTextureDescriptor(
      device.getLogicalDevice(), textureImageViews[mat.albedoTextureId].get(),
      textureImageViews[mat.normalTextureId].get(),
      textureImageViews[mat.metallicTextureId].get(),
      textureImageViews[mat.roughnessTextureId].get(),
      textureImageViews[mat.aoTextureId].get(), textureSampler);
  materialMap[materialKey] = mat.descriptorSetId;
  return mat;
}

int TextureManager::loadSkybox(const std::string &filename,
                               const VulkanDevice &device,
                               DescriptorManager & /*descriptorManager*/) {
  std::string key = normalizeTextureKey(filename);
  auto it = skyboxMap.find(key);
  if (it != skyboxMap.end())
    return it->second;

  int imageLoc = createHdrCubemap(key, device, 512);
  skyboxMap[key] = imageLoc;
  return imageLoc;
}

// ---------------------------------------------------------------------------
// IBL generation – delegates to GPU compute
// ---------------------------------------------------------------------------

int TextureManager::createIrradianceMap(int skyboxImageIndex,
                                        const VulkanDevice &device) {
  constexpr const char *cacheKey = "__irradiance_map__";
  auto it = textureMap.find(cacheKey);
  if (it != textureMap.end())
    return it->second;
  return dispatchIrradianceCompute(skyboxImageIndex, device);
}

int TextureManager::createPrefilteredEnvMap(int skyboxImageIndex,
                                            const VulkanDevice &device) {
  constexpr const char *cacheKey = "__prefiltered_env__";
  auto it = textureMap.find(cacheKey);
  if (it != textureMap.end())
    return it->second;
  return dispatchPrefilterCompute(skyboxImageIndex, device);
}

int TextureManager::loadBrdfLut(const VulkanDevice &device) {
  constexpr const char *cacheKey = "__brdf_lut__";
  auto it = textureMap.find(cacheKey);
  if (it != textureMap.end())
    return it->second;

  for (const char *rel :
       {"Resources/LUTs/ibl_brdf_lut.png", "LUTs/ibl_brdf_lut.png",
        "../Resources/LUTs/ibl_brdf_lut.png"}) {
    if (std::filesystem::exists(rel)) {
      int w, h, c;
      unsigned char *data = stbi_load(rel, &w, &h, &c, STBI_rgb_alpha);
      if (data) {
        int idx = createTextureImageFromPixels(cacheKey, data, w, h, device);
        stbi_image_free(data);
        return idx;
      }
    }
  }

  const uint32_t sz = 256;
  std::vector<uint8_t> pixels(sz * sz * 4);
  for (uint32_t r = 0; r < sz; r++) {
    float roughness = (r + 0.5f) / sz;
    float a = roughness * roughness;
    for (uint32_t n = 0; n < sz; n++) {
      float NdotV = (n + 0.5f) / sz;
      float k = a / 2.0f;
      float A = NdotV / (NdotV * (1.0f - k) + k);
      float fres = std::pow(1.0f - NdotV, 5.0f);
      float scale = A * (1.0f - fres);
      float bias = A * fres;
      size_t idx = (r * sz + n) * 4;
      pixels[idx + 0] =
          static_cast<uint8_t>(std::clamp(scale, 0.0f, 1.0f) * 255.0f);
      pixels[idx + 1] =
          static_cast<uint8_t>(std::clamp(bias, 0.0f, 1.0f) * 255.0f);
      pixels[idx + 2] = 0;
      pixels[idx + 3] = 255;
    }
  }
  return createTextureImageFromPixels(cacheKey, pixels.data(), sz, sz, device);
}

int TextureManager::createSsaoNoiseTexture(const VulkanDevice &device) {
  constexpr const char *cacheKey = "__ssao_noise__";
  auto it = textureMap.find(cacheKey);
  if (it != textureMap.end())
    return it->second;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::array<uint8_t, 4 * 4 * 4> pixels{};
  for (int i = 0; i < 16; i++) {
    float angle = dist(rng) * 2.0f * PI;
    pixels[i * 4 + 0] =
        static_cast<uint8_t>((std::cos(angle) * 0.5f + 0.5f) * 255.0f);
    pixels[i * 4 + 1] =
        static_cast<uint8_t>((std::sin(angle) * 0.5f + 0.5f) * 255.0f);
    pixels[i * 4 + 2] = 0;
    pixels[i * 4 + 3] = 255;
  }
  return createTextureImageFromPixels(cacheKey, pixels.data(), 4, 4, device,
                                      /*shareWithCompute=*/true);
}

// ---------------------------------------------------------------------------
// Private – samplers
// ---------------------------------------------------------------------------

void TextureManager::createTextureSampler(VkDevice device,
                                          VkPhysicalDevice physDevice) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDevice, &props);

  VkSamplerCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = VK_FILTER_LINEAR;
  ci.minFilter = VK_FILTER_LINEAR;
  ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  ci.anisotropyEnable = VK_TRUE;
  ci.maxAnisotropy = props.limits.maxSamplerAnisotropy;
  // Default maxLod is 0 -> only mip 0 sampled. Open the LOD range so the
  // generated mip chain is actually selectable.
  ci.minLod = 0.0f;
  ci.maxLod = VK_LOD_CLAMP_NONE;
  if (vkCreateSampler(device, &ci, nullptr, &textureSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create texture sampler");
}

void TextureManager::createSkyboxSampler(VkDevice device,
                                         VkPhysicalDevice physDevice) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDevice, &props);

  VkSamplerCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = VK_FILTER_LINEAR;
  ci.minFilter = VK_FILTER_LINEAR;
  ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  ci.minLod = 0.0f;
  ci.maxLod = static_cast<float>(IBL_PREFILTER_MIPS);
  ci.anisotropyEnable = VK_TRUE;
  ci.maxAnisotropy = props.limits.maxSamplerAnisotropy;
  if (vkCreateSampler(device, &ci, nullptr, &skyboxSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create skybox sampler");
}

// ---------------------------------------------------------------------------
// Private – compute pipeline
// ---------------------------------------------------------------------------

void TextureManager::createComputePipelines(const VulkanDevice &device,
                                            VkPipelineCache pipelineCache) {
  VkDevice dev = device.getLogicalDevice();

  std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 2;
  dslCI.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &computeSetLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create compute descriptor set layout");

  // Push constants used by prefilter (roughness + mip dimensions)
  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = sizeof(float) + sizeof(uint32_t) * 2;

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1;
  plCI.pSetLayouts = &computeSetLayout;
  plCI.pushConstantRangeCount = 1;
  plCI.pPushConstantRanges = &pcRange;
  if (vkCreatePipelineLayout(dev, &plCI, nullptr, &computeLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create compute pipeline layout");

  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8};
  poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8};
  VkDescriptorPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolCI.maxSets = 8;
  poolCI.poolSizeCount = 2;
  poolCI.pPoolSizes = poolSizes.data();
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &computePool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create compute descriptor pool");

  VkShaderModule irradianceMod = loadSpv(dev, "Shaders/irradiance.comp.spv");
  VkShaderModule prefilterMod = loadSpv(dev, "Shaders/prefilter.comp.spv");

  auto makeComputePipeline = [&](VkShaderModule module) -> VkPipeline {
    VkPipelineShaderStageCreateInfo stageCI = {};
    stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = module;
    stageCI.pName = "main";
    VkComputePipelineCreateInfo pCI = {};
    pCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pCI.stage = stageCI;
    pCI.layout = computeLayout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(dev, pipelineCache, 1, &pCI, nullptr,
                                 &pipeline) != VK_SUCCESS)
      throw std::runtime_error("Failed to create compute pipeline");
    return pipeline;
  };

  irradiancePipeline = makeComputePipeline(irradianceMod);
  prefilterPipeline = makeComputePipeline(prefilterMod);

  vkDestroyShaderModule(dev, irradianceMod, nullptr);
  vkDestroyShaderModule(dev, prefilterMod, nullptr);
}

void TextureManager::destroyComputePipelines(VkDevice dev) {
  if (computePool) {
    vkDestroyDescriptorPool(dev, computePool, nullptr);
    computePool = VK_NULL_HANDLE;
  }
  if (irradiancePipeline) {
    vkDestroyPipeline(dev, irradiancePipeline, nullptr);
    irradiancePipeline = VK_NULL_HANDLE;
  }
  if (prefilterPipeline) {
    vkDestroyPipeline(dev, prefilterPipeline, nullptr);
    prefilterPipeline = VK_NULL_HANDLE;
  }
  if (computeLayout) {
    vkDestroyPipelineLayout(dev, computeLayout, nullptr);
    computeLayout = VK_NULL_HANDLE;
  }
  if (computeSetLayout) {
    vkDestroyDescriptorSetLayout(dev, computeSetLayout, nullptr);
    computeSetLayout = VK_NULL_HANDLE;
  }
}

// ---------------------------------------------------------------------------
// Private – GPU IBL dispatch
// ---------------------------------------------------------------------------

int TextureManager::dispatchIrradianceCompute(int srcImageIndex,
                                              const VulkanDevice &device) {
  VkDevice dev = device.getLogicalDevice();
  const VkQueue computeQueue = device.getComputeQueue();
  const VkCommandPool computeCommandPool = device.getComputeCommandPool();
  constexpr uint32_t OUT_SIZE = 32;
  constexpr const char *cacheKey = "__irradiance_map__";
  const RenderQueueSharingInfo sharing =
      renderGraphicsComputeSharing(device.getQueueFamilies());

  VkImageCreateInfo imgCI = {};
  imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  imgCI.imageType = VK_IMAGE_TYPE_2D;
  imgCI.extent = {OUT_SIZE, OUT_SIZE, 1};
  imgCI.mipLevels = 1;
  imgCI.arrayLayers = 6;
  imgCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imgCI.samples = VK_SAMPLE_COUNT_1_BIT;
  applyRenderQueueSharing(imgCI, sharing);
  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &imgCI, &aci, &img, &alloc,
                     nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create irradiance cubemap image");
  AllocatedImage irradianceImg(device.getAllocator(), img, alloc);

  transitionCubemapLayout(
      dev, computeQueue, computeCommandPool, img, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL, 6);

  // 2D_ARRAY view for storage image write (compute sees 6 layers)
  VkImageViewCreateInfo avCI = {};
  avCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  avCI.image = img;
  avCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  avCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  avCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageView arrayView = VK_NULL_HANDLE;
  vkCreateImageView(dev, &avCI, nullptr, &arrayView);

  // Descriptor set
  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = computePool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &computeSetLayout;
  VkDescriptorSet ds = VK_NULL_HANDLE;
  vkAllocateDescriptorSets(dev, &dsAI, &ds);

  VkDescriptorImageInfo storageInfo = {};
  storageInfo.imageView = arrayView;
  storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorImageInfo srcInfo = {};
  srcInfo.sampler = skyboxSampler;
  srcInfo.imageView = textureImageViews[srcImageIndex].get();
  srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  std::array<VkWriteDescriptorSet, 2> writes{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ds;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[0].pImageInfo = &storageInfo;
  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = ds;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = &srcInfo;
  vkUpdateDescriptorSets(dev, 2, writes.data(), 0, nullptr);

  // Dispatch: each workgroup is 8×8×1 threads, cover 32×32×6 pixels
  VkCommandBuffer cmd = beginCommandBuffer(dev, computeCommandPool);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradiancePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout, 0,
                          1, &ds, 0, nullptr);
  uint32_t groups = (OUT_SIZE + 7) / 8;
  vkCmdDispatch(cmd, groups, groups, 6);

  // Barrier: compute writes done and the image is in shader-read layout before
  // the startup submit waits. This command buffer runs on the compute queue, so
  // both stage masks must be compute-compatible.
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = img;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  recordImageBarrier2(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);

  endAndSubmitCommandBuffer(dev, computeCommandPool, computeQueue, cmd);

  vkDestroyImageView(dev, arrayView, nullptr);
  vkFreeDescriptorSets(dev, computePool, 1, &ds);

  textureImages.push_back(std::move(irradianceImg));
  int loc = static_cast<int>(textureImages.size() - 1);

  VkImageViewCreateInfo cvCI = {};
  cvCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  cvCI.image = textureImages[loc].get();
  cvCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  cvCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  cvCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageView cubeView = VK_NULL_HANDLE;
  if (vkCreateImageView(dev, &cvCI, nullptr, &cubeView) != VK_SUCCESS)
    throw std::runtime_error("Failed to create irradiance cube view");
  textureImageViews.emplace_back(dev, cubeView);
  textureMap[cacheKey] = loc;
  return loc;
}

int TextureManager::dispatchPrefilterCompute(int srcImageIndex,
                                             const VulkanDevice &device) {
  VkDevice dev = device.getLogicalDevice();
  const VkQueue computeQueue = device.getComputeQueue();
  const VkCommandPool computeCommandPool = device.getComputeCommandPool();
  constexpr const char *cacheKey = "__prefiltered_env__";
  constexpr uint32_t BASE_SIZE = 128;
  const uint32_t nMips = static_cast<uint32_t>(IBL_PREFILTER_MIPS);
  const RenderQueueSharingInfo sharing =
      renderGraphicsComputeSharing(device.getQueueFamilies());

  VkImageCreateInfo imgCI = {};
  imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  imgCI.imageType = VK_IMAGE_TYPE_2D;
  imgCI.extent = {BASE_SIZE, BASE_SIZE, 1};
  imgCI.mipLevels = nMips;
  imgCI.arrayLayers = 6;
  imgCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imgCI.samples = VK_SAMPLE_COUNT_1_BIT;
  applyRenderQueueSharing(imgCI, sharing);
  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &imgCI, &aci, &img, &alloc,
                     nullptr) != VK_SUCCESS)
    throw std::runtime_error("Failed to create prefiltered env map image");
  AllocatedImage prefilterImg(device.getAllocator(), img, alloc);

  // All mip levels start as GENERAL so compute can write to them
  transitionCubemapLayout(
      dev, computeQueue, computeCommandPool, img, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL, 6, nMips);

  // One descriptor set, updated per mip
  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = computePool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &computeSetLayout;
  VkDescriptorSet ds = VK_NULL_HANDLE;
  vkAllocateDescriptorSets(dev, &dsAI, &ds);

  VkDescriptorImageInfo srcInfo = {};
  srcInfo.sampler = skyboxSampler;
  srcInfo.imageView = textureImageViews[srcImageIndex].get();
  srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  for (uint32_t mip = 0; mip < nMips; mip++) {
    uint32_t mipSize = BASE_SIZE >> mip;
    float roughness = static_cast<float>(mip) / static_cast<float>(nMips - 1);

    // Per-mip 2D_ARRAY view (compute writes to exactly this mip level)
    VkImageViewCreateInfo mvCI = {};
    mvCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    mvCI.image = img;
    mvCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    mvCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    mvCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
    VkImageView mipView = VK_NULL_HANDLE;
    vkCreateImageView(dev, &mvCI, nullptr, &mipView);

    VkDescriptorImageInfo storageInfo = {};
    storageInfo.imageView = mipView;
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ds;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &storageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ds;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &srcInfo;
    vkUpdateDescriptorSets(dev, 2, writes.data(), 0, nullptr);

    struct {
      float roughness;
      uint32_t w;
      uint32_t h;
    } pc{roughness, mipSize, mipSize};

    VkCommandBuffer cmd = beginCommandBuffer(dev, computeCommandPool);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout,
                            0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    uint32_t gx = (mipSize + 7) / 8;
    uint32_t gy = (mipSize + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 6);
    endAndSubmitCommandBuffer(dev, computeCommandPool, computeQueue, cmd);

    vkDestroyImageView(dev, mipView, nullptr);
  }

  // All mips written: transition to shader-read for fragment sampling
  transitionCubemapLayout(dev, computeQueue, computeCommandPool, img,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6, nMips);

  vkFreeDescriptorSets(dev, computePool, 1, &ds);

  textureImages.push_back(std::move(prefilterImg));
  int loc = static_cast<int>(textureImages.size() - 1);

  VkImageViewCreateInfo cvCI = {};
  cvCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  cvCI.image = textureImages[loc].get();
  cvCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  cvCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  cvCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, nMips, 0, 6};
  VkImageView cubeView = VK_NULL_HANDLE;
  if (vkCreateImageView(dev, &cvCI, nullptr, &cubeView) != VK_SUCCESS)
    throw std::runtime_error("Failed to create prefiltered env cube view");
  textureImageViews.emplace_back(dev, cubeView);
  textureMap[cacheKey] = loc;
  return loc;
}

// ---------------------------------------------------------------------------
// Private – texture creation helpers
// ---------------------------------------------------------------------------

int TextureManager::getOrCreateTexture(const std::string &filename,
                                       const VulkanDevice &device) {
  std::string norm = normalizeTextureKey(filename);
  // Cache identity must include the owning model's directory: file
  // resolution is modelBaseDirectory-relative, so two models shipping the
  // same bare name ("baseColor.png") are DIFFERENT textures. The bare
  // normalized name still drives the actual file load.
  std::string key = modelBaseDirectory + "|" + norm;
  auto it = textureMap.find(key);
  if (it != textureMap.end())
    return it->second;
  return createTextureImage(norm, key, device);
}

int TextureManager::getOrCreateFlatTexture(const std::string &cacheKey,
                                           uint8_t r, uint8_t g, uint8_t b,
                                           uint8_t a,
                                           const VulkanDevice &device) {
  auto it = textureMap.find(cacheKey);
  if (it != textureMap.end())
    return it->second;
  const unsigned char px[] = {r, g, b, a};
  return createTextureImageFromPixels(cacheKey, px, 1, 1, device);
}

int TextureManager::createTextureImage(const std::string &filename,
                                       const std::string &cacheKey,
                                       const VulkanDevice &device) {
  int w, h;
  VkDeviceSize sz;
  unsigned char *data = loadTextureFile(filename, &w, &h, &sz);
  int idx = createTextureImageFromPixels(cacheKey, data, w, h, device);
  stbi_image_free(data);
  return idx;
}

int TextureManager::createTextureImageFromPixels(const std::string &cacheKey,
                                                 const unsigned char *pixels,
                                                 int width, int height,
                                                 const VulkanDevice &device,
                                                 bool shareWithCompute) {
  // Mipmap chain: floor(log2(max(w,h))) + 1 levels. 1x1 textures (the flat
  // fallback colors) collapse to a single mip.
  const uint32_t mipLevels =
      static_cast<uint32_t>(
          std::floor(std::log2(static_cast<float>(std::max(width, height))))) +
      1u;

  // Build mip chain on the CPU. Box-filter every channel including alpha,
  // then rescale alpha per-mip with Castano coverage preservation so
  // alpha-tested cutouts (foliage) don't shrink and shimmer at distance.
  // Reference coverage is mip 0's at the engine's alpha-test threshold.
  std::vector<std::vector<uint8_t>> mipPixels(mipLevels);
  std::vector<std::pair<int, int>>   mipDims(mipLevels);

  size_t mip0Size = static_cast<size_t>(width) * height * 4;
  mipPixels[0].assign(pixels, pixels + mip0Size);
  mipDims[0] = {width, height};

  // Only run coverage preservation if mip 0 actually has alpha variation —
  // skips the work for opaque material textures (normals, roughness, etc.).
  bool hasAlpha = false;
  for (size_t i = 0; i < mip0Size && !hasAlpha; i += 4)
    if (mipPixels[0][i + 3] != 255) hasAlpha = true;

  float refCoverage = 0.0f;
  if (hasAlpha)
    refCoverage = computeAlphaCoverage(mipPixels[0], 1.0f, ALPHA_TEST_THRESHOLD);

  for (uint32_t m = 1; m < mipLevels; ++m) {
    int dw, dh;
    mipPixels[m] = boxDownsampleRGBA8(mipPixels[m - 1], mipDims[m - 1].first,
                                      mipDims[m - 1].second, dw, dh);
    mipDims[m] = {dw, dh};
    if (hasAlpha) {
      float scale = findAlphaScaleForCoverage(mipPixels[m], refCoverage,
                                              ALPHA_TEST_THRESHOLD);
      rescaleAlpha(mipPixels[m], scale);
    }
  }

  // Pack every mip into a single staging buffer at 4-byte-aligned offsets,
  // then issue one vkCmdCopyBufferToImage with mipLevels regions.
  std::vector<VkBufferImageCopy> regions(mipLevels);
  size_t stagingSize = 0;
  for (uint32_t m = 0; m < mipLevels; ++m) {
    regions[m] = {};
    regions[m].bufferOffset      = stagingSize;
    regions[m].imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
    regions[m].imageExtent       = {static_cast<uint32_t>(mipDims[m].first),
                                    static_cast<uint32_t>(mipDims[m].second), 1};
    stagingSize += mipPixels[m].size();
  }

  AllocatedBuffer staging;
  createBuffer(device.getAllocator(), stagingSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &staging);
  void *mapped;
  vmaMapMemory(device.getAllocator(), staging.getAllocation(), &mapped);
  for (uint32_t m = 0; m < mipLevels; ++m)
    std::memcpy(static_cast<char *>(mapped) + regions[m].bufferOffset,
                mipPixels[m].data(), mipPixels[m].size());
  vmaUnmapMemory(device.getAllocator(), staging.getAllocation());
  vmaFlushAllocation(device.getAllocator(), staging.getAllocation(), 0,
                     VK_WHOLE_SIZE);

  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  ci.mipLevels = mipLevels;
  ci.arrayLayers = 1;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  // Textures read by the dedicated compute queue (e.g. the SSAO noise tile)
  // need CONCURRENT sharing — EXCLUSIVE resources read on another family
  // without an ownership transfer have formally undefined contents.
  QueueFamilyIndices families = device.getQueueFamilies();
  uint32_t sharedFamilies[2] = {
      static_cast<uint32_t>(families.graphicsFamily),
      static_cast<uint32_t>(families.computeFamily)};
  if (shareWithCompute && device.hasDedicatedComputeQueue()) {
    ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = 2;
    ci.pQueueFamilyIndices = sharedFamilies;
  } else {
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &ci, &aci, &img, &alloc, nullptr) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create texture image");
  AllocatedImage texImg(device.getAllocator(), img, alloc);

  // Transition every mip level UNDEFINED → TRANSFER_DST.
  {
    VkCommandBuffer cmd = beginCommandBuffer(device.getLogicalDevice(),
                                             device.getGraphicsCommandPool());
    VkImageMemoryBarrier b = {};
    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.image         = texImg.get();
    b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    recordImageBarrier2(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, b);
    vkCmdCopyBufferToImage(cmd, staging.get(), texImg.get(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()),
                           regions.data());
    b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    recordImageBarrier2(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, b);
    endAndSubmitCommandBuffer(device.getLogicalDevice(),
                              device.getGraphicsCommandPool(),
                              device.getGraphicsQueue(), cmd);
  }

  textureImages.push_back(std::move(texImg));
  int loc = static_cast<int>(textureImages.size() - 1);

  VkImageViewCreateInfo vci = {};
  vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vci.image = textureImages[loc].get();
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.components = {
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};

  VkImageView view;
  if (vkCreateImageView(device.getLogicalDevice(), &vci, nullptr, &view) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create texture image view");
  textureImageViews.emplace_back(device.getLogicalDevice(), view);
  textureMap[cacheKey] = loc;
  return loc;
}

unsigned char *TextureManager::loadTextureFile(const std::string &filename,
                                               int *width, int *height,
                                               VkDeviceSize *imageSize) {
  int channels;
  std::string path = resolveTexturePath(filename);
  unsigned char *data =
      stbi_load(path.c_str(), width, height, &channels, STBI_rgb_alpha);
  if (!data)
    throw std::runtime_error("Failed to load texture: " + path);
  *imageSize = static_cast<VkDeviceSize>(*width) * (*height) * 4;
  return data;
}

std::string
TextureManager::resolveTexturePath(const std::string &filename) const {
  std::string norm = normalizeTextureKey(filename);
  std::filesystem::path input(norm);
  std::vector<std::filesystem::path> candidates = {
      input,
      std::filesystem::path("Textures") / input.filename(),
      std::filesystem::path("Models") / input,
      std::filesystem::path("Resources") / input,
      std::filesystem::path("Resources/Textures") / input,
      std::filesystem::path("Resources/HDRIs") / input.filename(),
      std::filesystem::path("Resources/LUTs") / input.filename(),
      std::filesystem::path("..") / input,
      std::filesystem::path("../Textures") / input.filename(),
      std::filesystem::path("../Resources") / input,
  };
  // Also search relative to the model file's directory (glTF textures live
  // there)
  if (!modelBaseDirectory.empty()) {
    std::filesystem::path mdir(modelBaseDirectory);
    candidates.push_back(mdir / input.filename());
    candidates.push_back(mdir / input);
  }
  for (const auto &c : candidates)
    if (std::filesystem::exists(c))
      return c.string();
  return candidates.front().string();
}

std::string
TextureManager::normalizeTextureKey(const std::string &filename) const {
  std::string key = filename;
  std::replace(key.begin(), key.end(), '\\', '/');
  constexpr const char *prefix = "textures/";
  if (key.size() > std::strlen(prefix) &&
      std::equal(
          prefix, prefix + std::strlen(prefix), key.begin(),
          [](char a, char b) { return std::tolower(a) == std::tolower(b); }))
    key = key.substr(std::strlen(prefix));
  return key;
}

// ---------------------------------------------------------------------------
// HDR cubemap
// ---------------------------------------------------------------------------

int TextureManager::createHdrCubemap(const std::string &filename,
                                     const VulkanDevice &device,
                                     uint32_t faceSize) {
  auto it = textureMap.find(filename);
  if (it != textureMap.end())
    return it->second;

  int w = 0, h = 0, ch = 0;
  std::string path = resolveTexturePath(filename);
  float *hdr = stbi_loadf(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
  if (!hdr)
    throw std::runtime_error("Failed to load HDR skybox: " + path);

  std::vector<float> cubemap(static_cast<size_t>(faceSize) * faceSize * 6 * 4);
  for (uint32_t face = 0; face < 6; face++) {
    for (uint32_t y = 0; y < faceSize; y++) {
      for (uint32_t x = 0; x < faceSize; x++) {
        float u = 2.0f * (x + 0.5f) / faceSize - 1.0f;
        float v = 2.0f * (y + 0.5f) / faceSize - 1.0f;
        glm::vec4 s =
            sampleEquirectangular(hdr, w, h, faceDirection(face, u, v));
        size_t dst =
            ((static_cast<size_t>(face) * faceSize + y) * faceSize + x) * 4;
        cubemap[dst + 0] = s.r;
        cubemap[dst + 1] = s.g;
        cubemap[dst + 2] = s.b;
        cubemap[dst + 3] = 1.0f;
      }
    }
  }
  stbi_image_free(hdr);

  VkDeviceSize imageSize = cubemap.size() * sizeof(float);
  const RenderQueueSharingInfo sharing =
      renderGraphicsComputeSharing(device.getQueueFamilies());
  AllocatedBuffer staging;
  createBuffer(device.getAllocator(), imageSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &staging);
  uploadToAllocation(device.getAllocator(), staging.getAllocation(),
                     cubemap.data(), imageSize);

  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.extent = {faceSize, faceSize, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 6;
  ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  applyRenderQueueSharing(ci, sharing);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage img = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  if (vmaCreateImage(device.getAllocator(), &ci, &aci, &img, &alloc, nullptr) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create HDR cubemap image");
  AllocatedImage cubemapImg(device.getAllocator(), img, alloc);

  transitionImageLayout(device.getLogicalDevice(), device.getGraphicsQueue(),
                        device.getGraphicsCommandPool(), cubemapImg.get(),
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6);
  copyImageBuffer(device.getLogicalDevice(), device.getGraphicsQueue(),
                  device.getGraphicsCommandPool(), staging.get(),
                  cubemapImg.get(), faceSize, faceSize, 6);
  transitionImageLayout(device.getLogicalDevice(), device.getGraphicsQueue(),
                        device.getGraphicsCommandPool(), cubemapImg.get(),
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6);

  textureImages.push_back(std::move(cubemapImg));
  int loc = static_cast<int>(textureImages.size() - 1);

  VkImageViewCreateInfo vci = {};
  vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vci.image = textureImages[loc].get();
  vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  vci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageView view;
  if (vkCreateImageView(device.getLogicalDevice(), &vci, nullptr, &view) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create HDR cubemap view");
  textureImageViews.emplace_back(device.getLogicalDevice(), view);
  textureMap[filename] = loc;
  return loc;
}
