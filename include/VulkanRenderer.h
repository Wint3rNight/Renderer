#pragma once
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
  void updateModel(glm::mat4 newModel);
  void cleanup();

  ~VulkanRenderer();

private:
  GLFWwindow *window;

  int currentFrame = 0;

  // scene objects
  std::vector<Mesh> mesheList;

  // scene settings
  struct MVP {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
  } mvp;

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

  // descriptors
  VkDescriptorSetLayout descriptorSetLayout;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBufferMemory;

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
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();

  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();

  void updateUniformBuffer(uint32_t imageIndex);

  // record functions
  void recordCommands();

  // get functions
  void getPhysicalDevice();
  int rateDeviceSuitability(VkPhysicalDevice device);

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
  VkShaderModule createShaderModule(const std::vector<char> &code);
};
