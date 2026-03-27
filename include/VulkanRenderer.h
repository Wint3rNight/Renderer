#pragma once
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "Mesh.h"
#include "Utilities.h"
#include "VulkanValidation.h"

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  void updateModel(int modelId, glm::mat4 newModel);
  void cleanup();

  ~VulkanRenderer();

private:
  GLFWwindow *window;

  int currentFrame = 0;

  // scene objects
  std::vector<Mesh> meshList;

  // scene settings
  struct UboViewProjection {
    glm::mat4 projection;
    glm::mat4 view;
  } uboViewProjection;

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
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkCommandBuffer>
      commandBuffers; // what the fuck was i going to write here

  VkImage depthBufferImage;
  VkDeviceMemory depthBufferImageMemory;
  VkImageView depthBufferImageView;

  // descriptors
  VkDescriptorSetLayout descriptorSetLayout;
  VkPushConstantRange pushConstantRange;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

  std::vector<VkBuffer> vpUniformBuffers;
  std::vector<VkDeviceMemory> vpUniformBufferMemory;

  std::vector<VkBuffer> modelDUniformBuffers;
  std::vector<VkDeviceMemory> modelDUniformBufferMemory;

  /*
    VkDeviceSize minUniformBufferOffset;
    size_t modelUniformAlignment;
   */
  // Model *modelTransferSpace;

  // pipeline
  VkPipeline graphicsPipeline;
  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;

  // pools
  VkCommandPool graphicsCommandPool;

  // utils
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  // syncronization
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight; // crash fix

  // vulkan functions
  // create functions
  void createInstance();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createRenderPass();
  void createDescriptorSetLayout();
  void createPushConstantRange();
  void createGraphicsPipeline();
  void createDepthBufferImage();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();

  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();

  void updateUniformBuffers(uint32_t imageIndex);

  // record functions
  void recordCommands(uint32_t currentImage);

  // get functions
  void getPhysicalDevice();
  int rateDeviceSuitability(VkPhysicalDevice device);

  // allocation functions
  void allocateDynamicBufferTransferSpace();

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
  VkFormat chooseSupportedFormat(const std::vector<VkFormat> &formats,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags featuresFlags);

  // support create functions
  VkImage createImage(uint32_t width, uint32_t height, VkFormat formats,
                      VkImageTiling tiling, VkImageUsageFlags useFlags,
                      VkMemoryPropertyFlags propFlags,
                      VkDeviceMemory *imageMemory);

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);

  VkShaderModule createShaderModule(const std::vector<char> &code);
};
