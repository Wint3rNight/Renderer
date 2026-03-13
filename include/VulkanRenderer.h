#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "Mesh.h"
#include "VulkanValidation.h"
#include "Utilities.h"

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  void cleanup();

  ~VulkanRenderer();

private:
  GLFWwindow *window;

  int currentFrame = 0;

  //scene objects
  Mesh firstMesh;

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
  std::vector<VkFence> imagesInFlight;//crash fix

  // vulkan functions
  // create functions
  void createInstance();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createRenderPass();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();

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
