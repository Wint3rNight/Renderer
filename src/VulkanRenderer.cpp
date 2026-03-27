#include "VulkanRenderer.h"
#include "Mesh.h"
#include "Utilities.h"
#include "VulkanValidation.h"

#include <GLFW/glfw3.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <vulkan/vulkan_core.h>

VulkanRenderer::VulkanRenderer() {}

// helper functions for debug messenger
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// helper function to destroy the debug messenger
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
    func(instance, debugMessenger, pAllocator);
}

int VulkanRenderer::init(GLFWwindow *newWindow) {
  window = newWindow;

  // create vulkan instance
  try {
    createInstance();
    createSurface();
    getPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createRenderPass();
    createDescriptorSetLayout();
    createPushConstantRange();
    createGraphicsPipeline();
    createDepthBufferImage();
    createFramebuffers();
    createCommandPool();

    uboViewProjection.projection = glm::perspective(
        glm::radians(45.0f),
        (float)swapChainExtent.width / (float)swapChainExtent.height, 0.1f,
        100.0f);

    uboViewProjection.view =
        glm::lookAt(glm::vec3(0.0f, .0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));

    uboViewProjection.projection[1][1] *=
        -1; // invert the y coordinate of the clip coordinates, because
            // vulkan has a different coordinate system than opengl

    // create a mesh(after creating the command pool ofc(i am a fucking
    // dumbass))

    // vertex data
    std::vector<Vertex> meshVertices = {
        {{-0.4f, 0.4f, 0.0f}, {1.0f, 0.0f, 0.0f}},  // 0
        {{-0.4f, -0.4f, 0.0f}, {1.0f, 0.0f, 0.0f}}, // 1
        {{0.4f, -0.4f, 0.0f}, {1.0f, 0.0f, 0.0f}},  // 2
        {{0.4f, 0.4f, 0.0f}, {1.0f, 0.0f, 0.0f}},   // 3
    };

    std::vector<Vertex> meshVertices2 = {
        {{-0.25f, 0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}},  // 0
        {{-0.25f, -0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}}, // 1
        {{0.25f, -0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}},  // 2
        {{0.25f, 0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}},   // 3
    };

    // index data
    std::vector<uint32_t> meshIndices = {0, 1, 2, 2, 3, 0};

    Mesh firstMesh =
        Mesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue,
             graphicsCommandPool, &meshVertices, &meshIndices);
    Mesh secondMesh =
        Mesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue,
             graphicsCommandPool, &meshVertices2, &meshIndices);

    meshList.push_back(firstMesh);
    meshList.push_back(secondMesh);

    createCommandBuffers();
    // allocateDynamicBufferTransferSpace();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();

    /* recordCommands(); */
    createSynchronization();

  } catch (const std::runtime_error &e) {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

void VulkanRenderer::updateModel(int modelId, glm::mat4 newModel) {
  if (modelId >= meshList.size()) {
    return;
  }
  meshList[modelId].setModel(newModel);
}

void VulkanRenderer::draw() {

  // wait for the fence to signal that command buffer has finished executing
  vkWaitForFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame],
                  VK_TRUE, std::numeric_limits<uint64_t>::max());
  // manually reset the fence to unsignaled state for the next frame
  /*  vkResetFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame]); */

  // acquire an image from the swap chain and signal when it is available
  uint32_t imageIndex;
  vkAcquireNextImageKHR(
      mainDevice.logicalDevice, swapchain, std::numeric_limits<uint64_t>::max(),
      imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
  if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) { // crash fix
    // If so, wait for the fence associated with that image
    vkWaitForFences(mainDevice.logicalDevice, 1, &imagesInFlight[imageIndex],
                    VK_TRUE, UINT64_MAX);
  }

  imagesInFlight[imageIndex] = drawFences[currentFrame];

  vkResetFences(mainDevice.logicalDevice, 1,
                &drawFences[currentFrame]); // crash fix

  recordCommands(imageIndex);

  updateUniformBuffers(imageIndex);
  // submit command buffer to graphics queue for execution
  // wait on the image to be available before starting execution, signal when
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1; // number of semaphores to wait on
  submitInfo.pWaitSemaphores =
      &imageAvailable[currentFrame]; // list of semaphores to wait on
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}; // pipeline stage to
                                                      // wait at for each
                                                      // semaphore in
                                                      // pWaitSemaphores
  submitInfo.pWaitDstStageMask =
      waitStages; // list of pipeline stages to wait at for
  // each semaphore in pWaitSemaphores
  submitInfo.commandBufferCount = 1; // number of command buffers to submit
  submitInfo.pCommandBuffers =
      &commandBuffers[imageIndex];     // list of command buffers to submit
  submitInfo.signalSemaphoreCount = 1; // number of semaphores to signal
  submitInfo.pSignalSemaphores =
      &renderFinished[currentFrame]; // list of semaphores to signal when
                                     // command buffers have finished execution
  // submit command buffer to graphics queue and execute
  VkResult result =
      vkQueueSubmit(graphicsQueue, 1, &submitInfo, drawFences[currentFrame]);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit draw command buffer");
  }

  // present the current image to the swap chain
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1; // number of semaphores to wait on
  presentInfo.pWaitSemaphores =
      &renderFinished[currentFrame]; // list of semaphores to wait on before
                                     // presentation
  presentInfo.swapchainCount = 1;    // number of swap chains to present to
  presentInfo.pSwapchains =
      &swapchain; // list of swap chains to present images to
  presentInfo.pImageIndices =
      &imageIndex; // list of image indices to present for each swap chai
  // present the image
  result = vkQueuePresentKHR(presentationQueue, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
  currentFrame =
      (currentFrame + 1) % MAX_FRAMES_DRAWS; // advance to the next frame
}

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(mainDevice.logicalDevice);

  // free(modelTransferSpace);
  /*
  added these in the cleanupSwapChain function hence removing coz it causes
  errors vkDestroyImageView(mainDevice.logicalDevice, depthBufferImageView,
  nullptr); vkDestroyImage(mainDevice.logicalDevice, depthBufferImage, nullptr);
    vkFreeMemory(mainDevice.logicalDevice, depthBufferImageMemory, nullptr); */

  vkDestroyDescriptorPool(mainDevice.logicalDevice, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, descriptorSetLayout,
                               nullptr);
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    vkDestroyBuffer(mainDevice.logicalDevice, vpUniformBuffers[i], nullptr);
    vkFreeMemory(mainDevice.logicalDevice, vpUniformBufferMemory[i], nullptr);
    /*   vkDestroyBuffer(mainDevice.logicalDevice, modelDUniformBuffers[i],
      nullptr); vkFreeMemory(mainDevice.logicalDevice,
      modelDUniformBufferMemory[i], nullptr); */
  }
  cleanupSwapChain();
  // wait for the logical device to finish operations before destroying
  // resources
  for (std::size_t i = 0; i < meshList.size(); i++) {
    meshList[i].destroyBuffers();
  }
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    vkDestroySemaphore(mainDevice.logicalDevice, renderFinished[i], nullptr);
    vkDestroySemaphore(mainDevice.logicalDevice, imageAvailable[i], nullptr);
    vkDestroyFence(mainDevice.logicalDevice, drawFences[i], nullptr);
  }
  vkDestroyCommandPool(mainDevice.logicalDevice, graphicsCommandPool, nullptr);

  for (auto framebuffer : swapChainFramebuffers) {
    vkDestroyFramebuffer(mainDevice.logicalDevice, framebuffer, nullptr);
  }

  vkDestroyPipeline(mainDevice.logicalDevice, graphicsPipeline, nullptr);

  vkDestroyPipelineLayout(mainDevice.logicalDevice, pipelineLayout, nullptr);

  vkDestroyRenderPass(mainDevice.logicalDevice, renderPass, nullptr);

  // destroy image views
  for (auto image : swapChainImages) {
    vkDestroyImageView(mainDevice.logicalDevice, image.imageView, nullptr);
  }

  // Destroy the Swap Chain
  vkDestroySwapchainKHR(mainDevice.logicalDevice, swapchain, nullptr);

  // Destroy the Logical Device first
  vkDestroyDevice(mainDevice.logicalDevice, nullptr);

  // Destroy the Surface
  vkDestroySurfaceKHR(instance, surface, nullptr);

  // Destroy the Debug Messenger
  if (enableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  }

  // destroy the Instance
  vkDestroyInstance(instance, nullptr);
}

VulkanRenderer::~VulkanRenderer() {}

void VulkanRenderer::createInstance() {
  // Check if the system/drivers actually have the validation layers
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("Validation layers requested, but not available!");
  }

  // info about the application
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan App";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  // create information for vkinstance
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // Get required extensions from GLFW
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> instanceExtensions(
      glfwExtensions, glfwExtensions + glfwExtensionCount);

  // Add the Debug Utils extension if validation is enabled
  //  This allows our custom callback to actually receive messages
  if (enableValidationLayers) {
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // Verify all requested extensions are supported by the GPU
  if (!checkInstanceExtensionSupport(&instanceExtensions)) {
    throw std::runtime_error("VkInstance does not support required extensions");
  }

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  // Hook the layers and the early debug messenger into creation
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // pNext part allows us to debug the actual creation and destruction
    // of the instance
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.pNext = nullptr;
  }

  // create the vulkan instance
  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  // 6. Setup the "permanent" debug messenger for the rest of the application
  if (enableValidationLayers) {
    if (CreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug messenger!");
    }
  }
}
void VulkanRenderer::createLogicalDevice() {
  // get the queue family indices for the physical device
  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);

  // vector for queue creation information
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  // set for family indices to avoid duplicate queue create info
  std::set<int> queueFamilyIndicies = {indices.graphicsFamily,
                                       indices.presentationFamily};

  // queues the logical device needs to create and info to do so
  for (int queueFamilyIndex : queueFamilyIndicies) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex =
        queueFamilyIndex; // index of the queue family to create a queue from
    queueCreateInfo.queueCount = 1; // number of queues to create
    float priority = 1.0f;
    queueCreateInfo.pQueuePriorities =
        &priority; // priority of the queues to create

    queueCreateInfos.push_back(queueCreateInfo);
  }

  // info about the logical device to create
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(
      queueCreateInfos.size()); // number of queue create infos
  deviceCreateInfo.pQueueCreateInfos =
      queueCreateInfos.data(); // list of queue create infos
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(
      deviceExtensions.size()); // number of device extensions to enable
  deviceCreateInfo.ppEnabledExtensionNames =
      deviceExtensions.data(); // list of enabled device extensions

  // physical device features
  VkPhysicalDeviceFeatures deviceFeatures = {};

  deviceCreateInfo.pEnabledFeatures =
      &deviceFeatures; // physical device features to use

  // create the logical device
  VkResult result = vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo,
                                   nullptr, &mainDevice.logicalDevice);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }

  // queues are created at the same time as the logical device
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0,
                   &graphicsQueue);
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.presentationFamily, 0,
                   &presentationQueue);
}

void VulkanRenderer::createSurface() {
  VkResult result =
      glfwCreateWindowSurface(instance, window, nullptr, &surface);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
}

void VulkanRenderer::createSwapChain() {
  // get swap chain details to pick best settings
  SwapChainDetails swapChainDetails =
      getSwapChainDetails(mainDevice.physicalDevice);

  // find the best settings for the swap chain
  VkSurfaceFormatKHR surfaceFormat =
      chooseBestSurfaceFormat(swapChainDetails.formats);
  VkPresentModeKHR presentationMode =
      chooseBestPresentationMode(swapChainDetails.presentModes);
  VkExtent2D extent = chooseSwapExtent(swapChainDetails.surfaceCapabilities);

  // no of images in swap chain, ask for 1 more than minimum to allow triple
  // buffering
  uint32_t imageCount = swapChainDetails.surfaceCapabilities.minImageCount + 1;

  // min image count must be less than the max image count (0 is max for no max)
  //  if max image count is 0, there is no maximum
  if (swapChainDetails.surfaceCapabilities.maxImageCount > 0 &&
      imageCount > swapChainDetails.surfaceCapabilities.maxImageCount) {
    imageCount = swapChainDetails.surfaceCapabilities.maxImageCount;
  }

  // create info for swap chain creation
  VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
  swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapChainCreateInfo.surface = surface;
  swapChainCreateInfo.imageFormat = surfaceFormat.format;
  swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
  swapChainCreateInfo.presentMode = presentationMode;
  swapChainCreateInfo.imageExtent = extent;
  swapChainCreateInfo.minImageCount = imageCount;
  swapChainCreateInfo.imageArrayLayers =
      1; // number of layers each image consists of
  swapChainCreateInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // what attachments images will be
                                           // used as

  swapChainCreateInfo.preTransform =
      swapChainDetails.surfaceCapabilities
          .currentTransform; // transform to perform on swapchain images
  swapChainCreateInfo.compositeAlpha =
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // alpha blending to use with other
                                         // windows
  swapChainCreateInfo.clipped =
      VK_TRUE; // whether to clip obscured pixels (by other windows)

  // get queue family indices to determine sharing mode
  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);

  // if graphics and presentation families are different, swapchain must be
  // shared between them which is not fast but can be done so why not
  if (indices.graphicsFamily != indices.presentationFamily) {
    // queue family indices to share between
    uint32_t queueFamilyIndices[] = {(uint32_t)indices.graphicsFamily,
                                     (uint32_t)indices.presentationFamily};
    swapChainCreateInfo.imageSharingMode =
        VK_SHARING_MODE_CONCURRENT; // images can be used across multiple queue
    swapChainCreateInfo.queueFamilyIndexCount =
        2; // number of queue families to share between
    swapChainCreateInfo.pQueueFamilyIndices =
        queueFamilyIndices; // list of queue families to share between

  } else {
    swapChainCreateInfo.imageSharingMode =
        VK_SHARING_MODE_EXCLUSIVE; // an image is owned by one queue family at a
                                   // time, best performance
    swapChainCreateInfo.queueFamilyIndexCount = 0;     // Optional
    swapChainCreateInfo.pQueueFamilyIndices = nullptr; // Optional
  }

  // if old swap chain is destroyed, we can use its resources for the new swap
  // chain
  swapChainCreateInfo.oldSwapchain =
      VK_NULL_HANDLE; // handle to old swap chain in case of recreation

  // create the swap chain
  VkResult result = vkCreateSwapchainKHR(
      mainDevice.logicalDevice, &swapChainCreateInfo, nullptr, &swapchain);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create swap chain");
  }

  // store swap chain images
  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = extent;

  // get swap chain images
  uint32_t swapChainImageCount;
  vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapchain,
                          &swapChainImageCount, nullptr);
  std::vector<VkImage> images(swapChainImageCount);
  vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapchain,
                          &swapChainImageCount, images.data());

  for (VkImage image : images) {
    // store image handle
    SwapChainImage swapChainImage = {};
    swapChainImage.image = image;

    // create image view
    swapChainImage.imageView =
        createImageView(image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

    // add swap chain image to list
    swapChainImages.push_back(swapChainImage);
  }
}

// crash fix
void VulkanRenderer::cleanupSwapChain() {

  for (auto framebuffer : swapChainFramebuffers) {
    vkDestroyFramebuffer(mainDevice.logicalDevice, framebuffer, nullptr);
  }
  swapChainFramebuffers.clear();

  for (auto image : swapChainImages) {
    vkDestroyImageView(mainDevice.logicalDevice, image.imageView, nullptr);
  }
  swapChainImages.clear();

  vkDestroyImageView(mainDevice.logicalDevice, depthBufferImageView, nullptr);
  vkDestroyImage(mainDevice.logicalDevice, depthBufferImage, nullptr);
  vkFreeMemory(mainDevice.logicalDevice, depthBufferImageMemory, nullptr);

  depthBufferImageView = VK_NULL_HANDLE;
  depthBufferImage = VK_NULL_HANDLE;
  depthBufferImageMemory = VK_NULL_HANDLE;

  if (swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(mainDevice.logicalDevice, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }
}

void VulkanRenderer::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }
  vkDeviceWaitIdle(mainDevice.logicalDevice);

  vkResetCommandPool(mainDevice.logicalDevice, graphicsCommandPool, 0);

  cleanupSwapChain();
  createSwapChain();
  createDepthBufferImage();

  imagesInFlight.assign(swapChainImages.size(), VK_NULL_HANDLE);

  createFramebuffers();
  createCommandBuffers();
}
// crash fox

void VulkanRenderer::createRenderPass() {
  // ATTACHMENTS
  // color attachment of render pass
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swapChainImageFormat;   // format of color attachment
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // number of samples to use
  colorAttachment.loadOp =
      VK_ATTACHMENT_LOAD_OP_CLEAR; // what to do with attachment before
                                   // rendering
  colorAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_STORE; // what to do with attachment after
                                    // rendering
  colorAttachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE; // what to do with stencil before
                                       // rendering (not using stencil)
  colorAttachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // what to do with stencil after
                                        // rendering (not using stencil)

  // framebuffer images will be stored as
  // an image but images can be given
  // different layout for optimal use, so
  // we need to specify the layout of the
  // image during different stages of the
  // render pass
  colorAttachment.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // layout of attachment before rendering(first
                                 // transition)
  colorAttachment.finalLayout =
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // layout of attachment after
                                       // rendering(layout of attachment when it
                                       // will be presented in the swap
                                       // chain(final transition))

  // Depth attachment of render pass
  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = chooseSupportedFormat(
      {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp =
      VK_ATTACHMENT_LOAD_OP_CLEAR; // clear depth at the start of the render
                                   // pass
  depthAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't need depth after render pass
  depthAttachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE; // we don't use stencil, so don't care
  depthAttachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't use stencil, so don't care
  depthAttachment.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // don't care about initial layout of depth
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // optimal layout for
                                                        // depth attachment

  // REFERENCES
  // attachment reference uses an attachment index to specify which attachment
  // to reference and the layout it will be in during a subpass
  VkAttachmentReference colorAttachmentReference = {};
  colorAttachmentReference.attachment = 0;
  colorAttachmentReference.layout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout of attachment during
                                                // subpass(second transition)

  VkAttachmentReference depthAttachmentReference = {};
  depthAttachmentReference.attachment = 1;
  depthAttachmentReference.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // layout of attachment
                                                        // during subpass(second
                                                        // transition)

  // information about the subpass, a subpass is a rendering pass that
  // references attachments, there can be multiple subpasses in a render pass
  // and they can reference the same attachments in different ways(eg. color
  // attachment in one subpass, input attachment in another subpass)
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint =
      VK_PIPELINE_BIND_POINT_GRAPHICS; // bind point of the subpass
  subpass.colorAttachmentCount = 1;    // number of color attachments
  subpass.pColorAttachments =
      &colorAttachmentReference; // list of color attachments
  subpass.pDepthStencilAttachment =
      &depthAttachmentReference; // depth attachment

  // determine subpass dependencies for layout transitions
  std::array<VkSubpassDependency, 2> subpassDependencies;
  // Conversion from VK_IMAGE_LAYOUT_UNDEFINED to
  // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  //  transition must happen after the render pass finishes and before the next
  //  render pass begins, so it must wait on the color attachment output stage
  //  of the first subpass to finish and must happen before the color attachment
  //  output stage of the second subpass begins
  //(after the previous render pass finishes)
  subpassDependencies[0].srcSubpass =
      VK_SUBPASS_EXTERNAL; // subpass index of source of dependency is external
                           // to the render pass
  subpassDependencies[0].srcStageMask =
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; // wait for all operations to be
  // finished
  subpassDependencies[0].srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT; // wait until memory is no longer being read

  // (before starting the next render pass)
  subpassDependencies[0].dstSubpass = 0; // our subpass is the destination
  subpassDependencies[0].dstStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // before starting color
                                                     // attachment output stage
  subpassDependencies[0].dstAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait until color attachment is no
  // longer being read from or written
  subpassDependencies[0].dependencyFlags = 0;

  // Conversion from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to
  // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR must happen after the color attachment
  // output stage of the subpass finishes and before the next render pass
  // begins, so it must wait on the color attachment output stage of the
  // first subpass to finish and must happen before the color attachment
  // output stage of the second subpass begins (after the previous render
  // pass finishes)
  subpassDependencies[1].srcSubpass = 0; // our subpass is the source
  subpassDependencies[1].srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait for color
                                                     // attachment output stage
  subpassDependencies[1].srcAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait until color attachment is no
                                            // longer being read from or written
  // before starting the next render pass
  subpassDependencies[1].dstSubpass =
      VK_SUBPASS_EXTERNAL; // subpass index of destination of dependency is
                           // external to the render pass
  subpassDependencies[1].dstStageMask =
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; // before starting the next render
                                            // pass
  subpassDependencies[1].dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT; // wait until memory is no longer being read
  subpassDependencies[1].dependencyFlags = 0;

  std::array<VkAttachmentDescription, 2> renderPassAttachments = {
      colorAttachment, depthAttachment};

  // create info for render pass creation
  VkRenderPassCreateInfo renderPassCreateInfo = {};
  renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(
      renderPassAttachments.size()); // number of attachments in render pass
  renderPassCreateInfo.pAttachments =
      renderPassAttachments.data();      // list of attachments in render pass
  renderPassCreateInfo.subpassCount = 1; // number of subpasses in render pass
  renderPassCreateInfo.pSubpasses =
      &subpass; // list of subpasses in render pass
  renderPassCreateInfo.dependencyCount =
      static_cast<uint32_t>(subpassDependencies.size()); // number of subpass
                                                         // dependencies
  renderPassCreateInfo.pDependencies =
      subpassDependencies.data(); // list of subpass dependencies

  VkResult result = vkCreateRenderPass(
      mainDevice.logicalDevice, &renderPassCreateInfo, nullptr, &renderPass);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create render pass");
  }
}

void VulkanRenderer::createDescriptorSetLayout() {
  // mvp binding info
  VkDescriptorSetLayoutBinding vpLayoutBinding = {};
  vpLayoutBinding.binding = 0; // binding number referenced in the shader
  vpLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of resource (uniform buffer)
  vpLayoutBinding.descriptorCount =
      1; // number of resources for binding, can be more than 1 for arrays
  vpLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT; // shader stage to bind the resources to
  vpLayoutBinding.pImmutableSamplers =
      nullptr; // used for image sampling, not used for uniform buffers

  // model binding info
  /*  VkDescriptorSetLayoutBinding modelLayoutBinding = {};
   modelLayoutBinding.binding = 1; // binding number referenced in the shader
   modelLayoutBinding.descriptorType =
       VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; // type of resource
   modelLayoutBinding.descriptorCount =
       1; // number of resources for binding, can be more than 1 for arrays
   modelLayoutBinding.stageFlags =
       VK_SHADER_STAGE_VERTEX_BIT; // shader stage to bind the resources to
   modelLayoutBinding.pImmutableSamplers =
       nullptr; // used for image sampling, not used for uniform buffers
  */
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {vpLayoutBinding};

  // create info for descriptor set layout creation
  VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
  layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCreateInfo.bindingCount = static_cast<uint32_t>(
      layoutBindings.size()); // number of bindings in the descriptor set
  layoutCreateInfo.pBindings =
      layoutBindings.data(); // list of bindings in the descriptor set

  // create the descriptor set layout
  VkResult result =
      vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &layoutCreateInfo,
                                  nullptr, &descriptorSetLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set layout");
  }
}

void VulkanRenderer::createPushConstantRange() {
  // define the push constant range(no creation needed because it is not a
  // vulkan object)
  pushConstantRange.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT; // shader stage to bind the push constant to
  pushConstantRange.offset = 0;   // offset of the push constant range
  pushConstantRange.size = sizeof(glm::mat4); // size of the push constant range
}

void VulkanRenderer::createGraphicsPipeline() {
  // read in spirv shader bytecode
  auto vertShaderCode = readFile("../Shaders/shader.vert.spv");
  auto fragShaderCode = readFile("../Shaders/shader.frag.spv");

  // create shader module
  VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  // vertex shader stage creation info
  VkPipelineShaderStageCreateInfo vertShaderCreateInfo = {};
  vertShaderCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // shader stage name
  vertShaderCreateInfo.module =
      vertShaderModule; // shader module containing code for shader stage
  vertShaderCreateInfo.pName = "main"; // entry point function

  // fragment shader stage creation info
  VkPipelineShaderStageCreateInfo fragShaderCreateInfo = {};
  fragShaderCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderCreateInfo.stage =
      VK_SHADER_STAGE_FRAGMENT_BIT; // shader stage name
  fragShaderCreateInfo.module =
      fragShaderModule; // shader module containing code for shader stage
  fragShaderCreateInfo.pName = "main"; // entry point function

  // putting shader stage create into into arrary because graphics pipeline
  // create info takes in an array of shader
  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderCreateInfo,
                                                    fragShaderCreateInfo};

  // how the data for a single vertex looks like as a whole
  VkVertexInputBindingDescription bindingDescription = {};
  bindingDescription.binding = 0; // can bind multiple streams of data
  bindingDescription.stride = sizeof(Vertex); // size of a single vertex
  bindingDescription.inputRate =
      VK_VERTEX_INPUT_RATE_VERTEX; // how to move to the next data entry after
                                   // each vertex
  // VK_VERTEX_INPUT_RATE_INSTANCE forinstancing
  // VK_VERTEX_INPUT_RATE_VERTEX to move to the next vertex
  // VK_VERTEX_INPUT_RATE_INDEX to move to the next index

  // how the data for the attributes is described
  std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
  // POSITION ATTRIBUTE
  attributeDescriptions[0].binding =
      0; // which binding the per vertex data comes from (should be same as
         // binding description binding)
  attributeDescriptions[0].location =
      0; // location in shader where data will be read from
  attributeDescriptions[0].format =
      VK_FORMAT_R32G32B32_SFLOAT; // format of the data (aslo size of the data)
  attributeDescriptions[0].offset =
      offsetof(Vertex, pos); // where the data is located in the vertex

  // COLOR ATTRIBUTE
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, col);

  // vertex input stage creation info
  VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
  vertexInputCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputCreateInfo.vertexBindingDescriptionCount =
      1; // number of vertex binding descriptions
  vertexInputCreateInfo.pVertexBindingDescriptions =
      &bindingDescription; // list of vertex binding descriptions
  vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(
      attributeDescriptions.size()); // number of vertex attribute descriptions
  vertexInputCreateInfo.pVertexAttributeDescriptions =
      attributeDescriptions.data(); // list of vertex attribute descriptions

  // input assembly stage creation info
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo = {};
  inputAssemblyCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyCreateInfo.topology =
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // primitive topology to assemble
                                           // vertices as
  inputAssemblyCreateInfo.primitiveRestartEnable =
      VK_FALSE; // allow overrider of strip topology to restart from a specific
                // vertex

  // viewport and scissor stage creation info
  VkViewport viewport = {};
  viewport.x = 0.0f; // x start point of the viewport
  viewport.y = 0.0f; // y start point of the viewport
  viewport.width = (float)swapChainExtent.width;   // width of the viewport
  viewport.height = (float)swapChainExtent.height; // height of the viewport
  viewport.minDepth = 0.0f;                        // min depth of the viewport
  viewport.maxDepth = 1.0f;                        // max depth of the viewport

  VkRect2D scissor = {};
  scissor.offset = {0, 0};          // offset of the scissor from the viewport
  scissor.extent = swapChainExtent; // extent of the scissor

  VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
  viewportStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportStateCreateInfo.viewportCount = 1;      // number of viewports
  viewportStateCreateInfo.pViewports = &viewport; // list of viewports
  viewportStateCreateInfo.scissorCount = 1;       // number of scissors
  viewportStateCreateInfo.pScissors = &scissor;   // list of scissors

  /*  // dynamic state stage creation info
   std::vector<VkDynamicState> dynamicStateEnables = {
       VK_DYNAMIC_STATE_VIEWPORT, // allow changing viewport without recreating
       VK_DYNAMIC_STATE_SCISSOR   // allow changing scissor without recreating
   };

   VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {};
   dynamicStateCreateInfo.sType =
       VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
   dynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(
       dynamicStateEnables.size()); // number of dynamic states
   dynamicStateCreateInfo.pDynamicStates =
       dynamicStateEnables.data(); // list of dynamic states */

  // rasterization stage creation info
  VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo = {};
  rasterizerCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizerCreateInfo.depthClampEnable =
      VK_FALSE; // allow clamping of depth values instead of discarding
  rasterizerCreateInfo.rasterizerDiscardEnable =
      VK_FALSE; // allow discarding of primitives before rasterization
  rasterizerCreateInfo.polygonMode =
      VK_POLYGON_MODE_FILL; // how to fill polygons (fill, line, point)
  rasterizerCreateInfo.lineWidth = 1.0f; // width of lines
  rasterizerCreateInfo.cullMode =
      VK_CULL_MODE_BACK_BIT; // which face of a polygon to cull
  rasterizerCreateInfo.frontFace =
      VK_FRONT_FACE_COUNTER_CLOCKWISE; // vertex order for front face
  rasterizerCreateInfo.depthBiasEnable =
      VK_FALSE; // allow depth bias for shadow mapping

  // multisampling stage creation info
  VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {};
  multisamplingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisamplingCreateInfo.sampleShadingEnable =
      VK_FALSE; // allow multisample shading for anti-aliasing
  multisamplingCreateInfo.rasterizationSamples =
      VK_SAMPLE_COUNT_1_BIT; // number of samples to use for multisampling per
                             // fragment

  // color blending stage creation info
  // blending is how the output color of the fragment shader is combined with
  // the color already in the framebuffer. Improtant for transparency,
  // other effects

  // blend attachment state creation info
  VkPipelineColorBlendAttachmentState colorState = {};
  colorState.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT |
      VK_COLOR_COMPONENT_A_BIT; // which color channels to write to
  colorState.blendEnable =
      VK_TRUE; // allow blending of the output color with the framebuffer color
  // blending equations for color blending : (srcColorBlendFactor * newColor)
  // colorBlendOp (dstColorBlendFactor * oldColor)
  colorState.srcColorBlendFactor =
      VK_BLEND_FACTOR_SRC_ALPHA; // blend factor for  new color
  colorState.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // blend factor for  old color
  colorState.colorBlendOp =
      VK_BLEND_OP_ADD; // blend operation to apply to the blended result

  // blending equations for alpha blending : (srcAlphaBlendFactor * newAlpha) +
  // (dstAlphaBlendFactor * oldAlpha)
  colorState.srcAlphaBlendFactor =
      VK_BLEND_FACTOR_ONE; // blend factor for the new alpha
  colorState.dstAlphaBlendFactor =
      VK_BLEND_FACTOR_ZERO; // blend factor for the old alpha
  colorState.alphaBlendOp =
      VK_BLEND_OP_ADD; // blend operation to apply to the blended alpha result

  VkPipelineColorBlendStateCreateInfo colorBlendingCreateInfo = {};
  colorBlendingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlendingCreateInfo.logicOpEnable =
      VK_FALSE; // allow use of logical operations for blending
  colorBlendingCreateInfo.attachmentCount = 1; // number of color attachments
  colorBlendingCreateInfo.pAttachments =
      &colorState; // list of color blend attachment states

  // pipeline layout creation info
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount =
      1; // number of descriptor sets to be used in the pipeline
  pipelineLayoutCreateInfo.pSetLayouts =
      &descriptorSetLayout; // list of descriptor set layouts to be used in the
                            // pipeline
  pipelineLayoutCreateInfo.pushConstantRangeCount =
      1; // number of push constant ranges to be used in the pipeline
  pipelineLayoutCreateInfo.pPushConstantRanges =
      &pushConstantRange; // list of push constant ranges to be used in the
                          // pipeline

  // create pipeline layout
  VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice,
                                           &pipelineLayoutCreateInfo, nullptr,
                                           &pipelineLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }

  // depth and stencil testing
  VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo = {};
  depthStencilCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilCreateInfo.depthTestEnable =
      VK_TRUE; // allow depth testing for fragments
  depthStencilCreateInfo.depthWriteEnable =
      VK_TRUE; // allow writing to the depth buffer
  depthStencilCreateInfo.depthCompareOp =
      VK_COMPARE_OP_LESS; // comparison operation for depth testing
  depthStencilCreateInfo.depthBoundsTestEnable =
      VK_FALSE; // allow depth bounds testing
  depthStencilCreateInfo.stencilTestEnable = VK_FALSE; // allow stencil testing

  // graphics pipeline create info
  VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stageCount = 2;         // number of shader stages
  pipelineCreateInfo.pStages = shaderStages; // list of shader stages
  pipelineCreateInfo.pVertexInputState =
      &vertexInputCreateInfo; // vertex input stage create info
  pipelineCreateInfo.pInputAssemblyState =
      &inputAssemblyCreateInfo; // input assembly stage create info
  pipelineCreateInfo.pViewportState =
      &viewportStateCreateInfo; // viewport and scissor stage create info
  pipelineCreateInfo.pDynamicState = nullptr; // dynamic state
  pipelineCreateInfo.pRasterizationState =
      &rasterizerCreateInfo; // rasterization stage create info
  pipelineCreateInfo.pMultisampleState =
      &multisamplingCreateInfo; // multisampling stage create info
  pipelineCreateInfo.pDepthStencilState =
      &depthStencilCreateInfo; // depth and stencil state create info
  pipelineCreateInfo.pColorBlendState =
      &colorBlendingCreateInfo; // color blending stage create info
  pipelineCreateInfo.layout =
      pipelineLayout; // pipeline layout to use for pipeline
  pipelineCreateInfo.renderPass = renderPass; // render pass to use for pipeline
  pipelineCreateInfo.subpass =
      0; // subpass index of render pass to use for pipeline

  // pipeline derivatives : can create multiple pipelines that derive from a
  // base pipeline to save time during pipeline creation, if base pipeline is
  // not yet created, can use pipeline index to reference it
  pipelineCreateInfo.basePipelineHandle =
      VK_NULL_HANDLE; // handle of an existing pipeline to derive from
  pipelineCreateInfo.basePipelineIndex =
      -1; // index of an existing pipeline in the same pipeline cache to derive
          // from

  // create graphics pipeline
  result = vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE,
                                     1, &pipelineCreateInfo, nullptr,
                                     &graphicsPipeline);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics pipeline");
  }

  // destroy shader modules that are no long needed
  vkDestroyShaderModule(mainDevice.logicalDevice, fragShaderModule, nullptr);
  vkDestroyShaderModule(mainDevice.logicalDevice, vertShaderModule, nullptr);
}

void VulkanRenderer::createDepthBufferImage() {
  // get supported depth format for depth buffer image
  VkFormat depthFormat = chooseSupportedFormat(
      {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

  // create depth buffer image and its memory
  depthBufferImage = createImage(
      swapChainExtent.width, swapChainExtent.height, depthFormat,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &depthBufferImageMemory);

  // create depth buffer image view
  depthBufferImageView =
      createImageView(depthBufferImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VulkanRenderer::createFramebuffers() {
  swapChainFramebuffers.resize(
      swapChainImages.size()); // resize fb list to no.of swapchain images
                               // create framebuffer for each swap chain image
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    std::array<VkImageView, 2> attachments = {swapChainImages[i].imageView,
                                              depthBufferImageView};

    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.renderPass = renderPass; // render pass to use
    framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(
        attachments.size()); // number of attachments in framebuffer (only color
                             // attachment)
    framebufferCreateInfo.pAttachments =
        attachments.data(); // list of attachments(1:1 with render pass)
    framebufferCreateInfo.width = swapChainExtent.width;
    framebufferCreateInfo.height = swapChainExtent.height;
    framebufferCreateInfo.layers =
        1; // number of layers in framebuffer (1 unless doing stereoscopic 3D)

    VkResult result =
        vkCreateFramebuffer(mainDevice.logicalDevice, &framebufferCreateInfo,
                            nullptr, &swapChainFramebuffers[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create framebuffer");
    }
  }
}

void VulkanRenderer::createCommandPool() {
  // get indices of queue families to create command pool for
  QueueFamilyIndices queueFamilyIndices =
      getQueueFamilies(mainDevice.physicalDevice);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex =
      queueFamilyIndices.graphicsFamily; // command buffers from this pool will
                                         // be submitted to this queue family
  // create graphics queue family command pool
  VkResult result = vkCreateCommandPool(mainDevice.logicalDevice, &poolInfo,
                                        nullptr, &graphicsCommandPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool");
  }
}

void VulkanRenderer::createCommandBuffers() {
  commandBuffers.resize(
      swapChainFramebuffers.size()); // resize cb list to no. of framebuffers

  VkCommandBufferAllocateInfo cbAllocInfo = {};
  cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAllocInfo.commandPool =
      graphicsCommandPool; // command pool to allocate from
  cbAllocInfo.level =
      VK_COMMAND_BUFFER_LEVEL_PRIMARY; // level of the command buffers (primary
                                       // can be submitted, secondary
                                       // cannot(primary executed by queues,
                                       // secondary executed by primary))
  cbAllocInfo.commandBufferCount = static_cast<uint32_t>(
      commandBuffers.size()); // number of
                              // command buffers to allocate
  // allocate command buffers and place handles in array of buffers
  VkResult result = vkAllocateCommandBuffers(
      mainDevice.logicalDevice, &cbAllocInfo, commandBuffers.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffers");
  }
}

void VulkanRenderer::createSynchronization() {
  // resize synchronization primitive arrays to max number of frames that can be
  // processed concurrently
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  renderFinished.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);

  imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE); // crash fix

  // semaphore creation info
  VkSemaphoreCreateInfo semaphoreCreateInfo = {};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  // fence creation info
  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags =
      VK_FENCE_CREATE_SIGNALED_BIT; // create in signaled state so that first
                                    // frame can be rendered without waiting

  // create semaphores
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo,
                          nullptr, &imageAvailable[i]) != VK_SUCCESS ||
        vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo,
                          nullptr, &renderFinished[i]) != VK_SUCCESS ||
        vkCreateFence(mainDevice.logicalDevice, &fenceCreateInfo, nullptr,
                      &drawFences[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create synchronization primitives");
    }
  }
}

void VulkanRenderer::createUniformBuffers() {
  // view projection buffer size
  VkDeviceSize vpBufferSize = sizeof(UboViewProjection);

  // model buffer size
  /* VkDeviceSize modelBufferSize =
      modelUniformAlignment *
      MAX_OBJECTS; // size of model uniform buffer, must be large enough to hold
                   // the data for all objects in the scene
 */
  vpUniformBuffers.resize(swapChainImages.size()); // create a uniform buffer
                                                   // for each swap chain image
  vpUniformBufferMemory.resize(
      swapChainImages.size()); // create memory for each uniform buffer
                               /*
                                 modelDUniformBuffers.resize(
                                     swapChainImages.size()); // create a uniform buffer for
                                                              // each swap chain image
                                 modelDUniformBufferMemory.resize(
                                     swapChainImages.size()); // create memory for each uniform buffer
                                */
  // create uniform buffers and allocate memory for them
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice,
                 vpBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &vpUniformBuffers[i], &vpUniformBufferMemory[i]);
  }

  /*   for (size_t i = 0; i < swapChainImages.size(); i++) {
      createBuffer(mainDevice.physicalDevice, mainDevice.logicalDevice,
                   modelBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   &modelDUniformBuffers[i], &modelDUniformBufferMemory[i]);
    } */
}

void VulkanRenderer::createDescriptorPool() {
  // type of descriptor and number of descriptors to create in pool(and not
  // descriptor sets(shi to remember yk))
  VkDescriptorPoolSize vpPoolSize = {};
  vpPoolSize.type =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of resource in pool
  vpPoolSize.descriptorCount = static_cast<uint32_t>(
      vpUniformBuffers.size()); // number of descriptors in pool

  // model buffer pool size info
  /*   VkDescriptorPoolSize modelDPoolSize = {};
    modelDPoolSize.type =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; // type of resource in pool
    modelDPoolSize.descriptorCount = static_cast<uint32_t>(
        modelDUniformBuffers.size()); // number of descriptors in pool
   */
  // list of pool sizes
  std::vector<VkDescriptorPoolSize> descriptorPoolSizes = {vpPoolSize};

  // create info for descriptor pool creation
  VkDescriptorPoolCreateInfo poolCreateInfo = {};
  poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCreateInfo.maxSets = static_cast<uint32_t>(
      swapChainImages.size()); // max number of descriptor
                               // sets that can be allocateduniform
  poolCreateInfo.poolSizeCount = static_cast<uint32_t>(
      descriptorPoolSizes.size()); // number of descriptor types in pool
  poolCreateInfo.pPoolSizes =
      descriptorPoolSizes.data(); // list of descriptor types and counts in pool

  // create descriptor pool
  VkResult result = vkCreateDescriptorPool(
      mainDevice.logicalDevice, &poolCreateInfo, nullptr, &descriptorPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }
}

void VulkanRenderer::createDescriptorSets() {
  descriptorSets.resize(swapChainImages.size()); // create a descriptor set for
  // each swap chain image

  std::vector<VkDescriptorSetLayout> setLayouts(
      swapChainImages.size(), descriptorSetLayout); // list of descriptor set
                                                    // layouts to use for each
                                                    // allocated descriptor set

  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool =
      descriptorPool; // descriptor pool to allocate from
  setAllocInfo.descriptorSetCount = static_cast<uint32_t>(
      swapChainImages.size()); // number of descriptor sets to allocate
  setAllocInfo.pSetLayouts =
      setLayouts.data(); // layout to use for each allocated descriptor set

  // allocate descriptor sets, multiple
  VkResult result = vkAllocateDescriptorSets(
      mainDevice.logicalDevice, &setAllocInfo, descriptorSets.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor sets");
  }

  // update descriptor sets with buffer info for each swap chain image
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    // view projection descriptor
    //  buffer inof and data offset info
    VkDescriptorBufferInfo vpBufferInfo = {};
    vpBufferInfo.buffer = vpUniformBuffers[i]; // buffer to bind to descriptor
    vpBufferInfo.offset = 0;                   // offset of data in buffer
    vpBufferInfo.range = sizeof(UboViewProjection); // size of data in buffer

    // data about connection bwtn descriptor set and buffer info to update
    // descriptor set with
    VkWriteDescriptorSet vpSetWrite = {};
    vpSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vpSetWrite.dstSet = descriptorSets[i]; // descriptor set to update
    vpSetWrite.dstBinding = 0; // binding of the descriptor to update (should
                               // match binding in shader.veryt)
    vpSetWrite.dstArrayElement = 0; // first index in array to update
    vpSetWrite.descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of descriptor
    vpSetWrite.descriptorCount =
        1; // number of descriptors to update (size of array)
    vpSetWrite.pBufferInfo =
        &vpBufferInfo; // buffer info to write to descriptor

    // model descriptor
    /* VkDescriptorBufferInfo modelDBufferInfo = {};
    modelDBufferInfo.buffer =
        modelDUniformBuffers[i]; // buffer to bind to descriptor
    modelDBufferInfo.offset = 0; // offset of data in buffer
    modelDBufferInfo.range = modelUniformAlignment; // size of data in buffer

    VkWriteDescriptorSet modelSetWrite = {};
    modelSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    modelSetWrite.dstSet = descriptorSets[i]; // descriptor set to update
    modelSetWrite.dstBinding = 1;      // binding of the descriptor to update
                                       // (should match binding in shader.veryt)
    modelSetWrite.dstArrayElement = 0; // first index in array to update
    modelSetWrite.descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; // type of descriptor
    modelSetWrite.descriptorCount =
        1; // number of descriptors to update (size of array)
    modelSetWrite.pBufferInfo =
        &modelDBufferInfo; // buffer info to write to descriptor
 */
    // list of descriptor set writes to update
    std::vector<VkWriteDescriptorSet> setWrites = {vpSetWrite};

    // update the descriptor set with new buffer info
    vkUpdateDescriptorSets(mainDevice.logicalDevice,
                           static_cast<uint32_t>(setWrites.size()),
                           setWrites.data(), 0, nullptr);
  }
}

void VulkanRenderer::updateUniformBuffers(uint32_t imageIndex) {
  // copy vp data to uniform buffer
  void *data;
  vkMapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex], 0,
              sizeof(UboViewProjection), 0, &data);
  memcpy(data, &uboViewProjection, sizeof(UboViewProjection));
  vkUnmapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex]);

  // copy model data to uniform buffer
  /* for (size_t i = 0; i < mesheList.size(); i++) {
    Model *thisModel =
        (Model *)((uint64_t)modelTransferSpace + (i * modelUniformAlignment));
    *thisModel = mesheList[i].getModel();
  }

  // map memory and copy data for all models at once since they are in
  // contigious memory then unmap memory
  vkMapMemory(mainDevice.logicalDevice, modelDUniformBufferMemory[imageIndex],
              0, modelUniformAlignment * mesheList.size(), 0, &data);
  memcpy(data, modelTransferSpace, modelUniformAlignment * mesheList.size());
  vkUnmapMemory(mainDevice.logicalDevice,
                modelDUniformBufferMemory[imageIndex]); */
}

void VulkanRenderer::recordCommands(uint32_t currentImage) {
  vkResetCommandBuffer(commandBuffers[currentImage], 0);

  // info about the command buffer to begin recording commands into
  VkCommandBufferBeginInfo bufferBeginInfo = {};
  bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  // infoo about the render pass to begin for the command buffer
  VkRenderPassBeginInfo renderPassBeginInfo = {};
  renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBeginInfo.renderPass = renderPass; // render pass to begin
  renderPassBeginInfo.renderArea.offset = {
      0, 0}; // offset of render area from the framebuffer
  renderPassBeginInfo.renderArea.extent =
      swapChainExtent; // extent of render area (starting from offset)

  std::array<VkClearValue, 2> clearValues = {};
  clearValues[0].color = {0.6f, 0.65f, 0.4f, 1.0f};
  clearValues[1].depthStencil.depth = 1.0f; // depth to clear depth buffer to
                                            // and stencil to clear stencil
                                            // buffer to
  renderPassBeginInfo.pClearValues = clearValues.data(); // list of clear values
  renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(
      clearValues.size()); // number of clear values in list
                           // (same as number of attachments)

  renderPassBeginInfo.framebuffer =
      swapChainFramebuffers[currentImage]; // framebuffer to use for render pass
  // begin recording commands into command buffer
  VkResult result =
      vkBeginCommandBuffer(commandBuffers[currentImage], &bufferBeginInfo);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to begin recording command buffer");
  }

  vkCmdBeginRenderPass(commandBuffers[currentImage], &renderPassBeginInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  // bind the graphics pipeline so that it will be used for rendering
  vkCmdBindPipeline(commandBuffers[currentImage],
                    VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
  for (std::size_t j = 0; j < meshList.size(); j++) {
    VkBuffer vertexBuffers[] = {
        meshList[j].getVertexBuffer()}; // buffers to bind
    VkDeviceSize offsets[] = {0};       // offsets into buffers to start from
    vkCmdBindVertexBuffers(commandBuffers[currentImage], 0, 1, vertexBuffers,
                           offsets); // command to bind vertex buffers (also
                                     // binds the vertex input
    // state specified during pipeline creation)

    // bind index buffer
    vkCmdBindIndexBuffer(commandBuffers[currentImage],
                         meshList[j].getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

    // dynamoc offset amount
    // uint32_t dynamicOffset = static_cast<uint32_t>(modelUniformAlignment) *
    // j;

    // push constant for model data
    vkCmdPushConstants(commandBuffers[currentImage], pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model),
                       &meshList[j].getModel());

    // bind descriptor sets (for MVP matrix)
    vkCmdBindDescriptorSets(commandBuffers[currentImage],
                            VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                            1, &descriptorSets[currentImage], 0, nullptr);

    // ececute a draw command with the pipeline
    vkCmdDrawIndexed(commandBuffers[currentImage], meshList[j].getIndexCount(),
                     1, 0, 0, 0);
  }

  vkCmdEndRenderPass(commandBuffers[currentImage]);

  // stop recording commands into command buffer
  result = vkEndCommandBuffer(commandBuffers[currentImage]);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to stop recording command buffer");
  }
}

int VulkanRenderer::rateDeviceSuitability(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  int score = 0;
  // Discrete GPUs have a significant performance advantage
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000;
  }
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    score += 100;
  }
  return score;
}

void VulkanRenderer::getPhysicalDevice() {
  // enumerate physical devices the vkinstance can access
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("Failed to find a GPU with Vulkan support");
  }

  // get list of physical devices
  std::vector<VkPhysicalDevice> deviceList(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, deviceList.data());

  VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
  int highestScore = -1;
  for (const auto &device : deviceList) {
    if (checkDeviceSuitable(device)) {
      int score = rateDeviceSuitability(device);
      if (score > highestScore) {
        bestDevice = device;
        highestScore = score;
      }
    }
  }

  if (bestDevice != VK_NULL_HANDLE) {
    mainDevice.physicalDevice = bestDevice;
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(mainDevice.physicalDevice, &deviceProperties);
    /*
        minUniformBufferOffset =
            deviceProperties.limits.minUniformBufferOffsetAlignment; */

    printf("Selected GPU: %s\n", deviceProperties.deviceName);
    printf("Device Type: %d\n", deviceProperties.deviceType);
    printf("Total Devices Found %d\n", deviceCount);
    /* printf("Min Uniform Buffer Offset Alignment: %llu\n",
           static_cast<unsigned long long>(minUniformBufferOffset)); */
    printf("API Version: %d.%d.%d\n",
           VK_VERSION_MAJOR(deviceProperties.apiVersion),
           VK_VERSION_MINOR(deviceProperties.apiVersion),
           VK_VERSION_PATCH(deviceProperties.apiVersion));
  } else {
    throw std::runtime_error("Failed to find a suitable GPU");
  }
}

void VulkanRenderer::allocateDynamicBufferTransferSpace() {
  /*  // calculate dynamic alignment of model uniform buffer
   modelUniformAlignment = (sizeof(Model) + minUniformBufferOffset - 1) &
                           ~(minUniformBufferOffset - 1);

   size_t totalSize = modelUniformAlignment * MAX_OBJECTS;

   // creat space in memory to hold the dynamic uniform buffer data to be
   // transferred to the GPU
   modelTransferSpace = (Model *)aligned_alloc(modelUniformAlignment,
   totalSize);

   if (modelTransferSpace == nullptr) {
     throw std::runtime_error("Failed to allocate aligned transfer space!");
   } */
}

bool VulkanRenderer::checkInstanceExtensionSupport(
    std::vector<const char *> *checkEtensions) {

  // get all available extensions
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  // create a list of all vkextension properties
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                         extensions.data());

  // check if all extensions in checkExtensions are in the list of available
  for (const auto &checkExtension : *checkEtensions) {
    bool hasExtension = false;

    for (const auto &extension : extensions) {
      if (strcmp(checkExtension, extension.extensionName) == 0) {
        hasExtension = true;
        break;
      }
    }
    if (!hasExtension) {
      return false;
    }
  }
  return true;
}

bool VulkanRenderer::checkDeviceSuitable(VkPhysicalDevice device) {
  /*
    // infor about the device
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures); */

  QueueFamilyIndices indices = getQueueFamilies(device);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainValid = false;

  if (extensionsSupported) {
    SwapChainDetails swapChainDetails = getSwapChainDetails(device);
    swapChainValid = !swapChainDetails.formats.empty() &&
                     !swapChainDetails.presentModes.empty();
  }

  return indices.isValid() && extensionsSupported && swapChainValid;
}

QueueFamilyIndices VulkanRenderer::getQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  // get all the queue families of the device
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyList(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilyList.data());

  // go through each queue family and check if it has the required queues
  int i = 0;

  for (const auto &queueFamily : queueFamilyList) {
    // check if queue family has graphics capabilities
    if (queueFamily.queueCount > 0 &&
        queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i; // set graphics family index
    }

    // check if queue family supporst  presentation
    VkBool32 presentationSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                         &presentationSupport);
    if (queueFamily.queueCount > 0 && presentationSupport) {
      indices.presentationFamily = i; // set presentation family index
    }

    // check if queue family is valid, if so break loop
    if (indices.isValid()) {
      break;
    }

    i++;
  }
  return indices;
}

bool VulkanRenderer::checkDeviceExtensionSupport(VkPhysicalDevice device) {
  // get device extension count
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  if (extensionCount == 0) {
    return false;
  }
  // populate list of extensions
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       extensions.data());
  // check for extensions
  for (const auto &deviceExtension : deviceExtensions) {
    bool hasExtension = false;
    for (const auto &extension : extensions) {
      if (strcmp(deviceExtension, extension.extensionName) == 0) {
        hasExtension = true;
        break;
      }
    }
    if (!hasExtension) {
      return false;
    }
  }
  return true;
}

bool VulkanRenderer::checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  // getting all the layers
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  // check for availability
  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound)
      return false;
  }

  return true;
}

SwapChainDetails VulkanRenderer::getSwapChainDetails(VkPhysicalDevice device) {
  SwapChainDetails swapChainDetails;

  // capabilities
  // getting the surface capabilities of the device and surface
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      device, surface, &swapChainDetails.surfaceCapabilities);

  // formats
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  // if format count is not zero, get list of surface formats
  if (formatCount != 0) {
    swapChainDetails.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         swapChainDetails.formats.data());
  }

  // presentation modes
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);

  // if presentation mode count is not zero, get list of presentation modes
  if (presentModeCount != 0) {
    swapChainDetails.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount,
        swapChainDetails.presentModes.data());
  }

  return swapChainDetails;
}

VkSurfaceFormatKHR VulkanRenderer::chooseBestSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &formats) {
  // if only 1 format is available and is undefined, means all formats are
  // available
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  // look for optimal surface format
  for (const auto &format : formats) {
    if ((format.format == VK_FORMAT_R8G8B8A8_UNORM ||
         format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }

  // if optimal format isn't found, return first format available
  return formats[0];
}

VkPresentModeKHR VulkanRenderer::chooseBestPresentationMode(
    const std::vector<VkPresentModeKHR> &presentationModes) {
  // look for mailbox presentation mode, best for triple buffering, low
  // latency
  for (const auto &presentationMode : presentationModes) {
    if (presentationMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return presentationMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRenderer::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
  // if current extent is max, then extent can vary, so we set it to the
  // window size
  if (surfaceCapabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return surfaceCapabilities.currentExtent;
  } else {
    // if value vary, set manually

    // get window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // create new extent using window size
    VkExtent2D newExtent = {};
    newExtent.width = static_cast<uint32_t>(width);
    newExtent.height = static_cast<uint32_t>(height);

    // surface also defines min and max image extents so make sure the new
    // extent is within bounds
    newExtent.width = std::max(
        surfaceCapabilities.minImageExtent.width,
        std::min(surfaceCapabilities.maxImageExtent.width, newExtent.width));
    newExtent.height = std::max(
        surfaceCapabilities.minImageExtent.height,
        std::min(surfaceCapabilities.maxImageExtent.height, newExtent.height));
    return newExtent;
  }
}

VkFormat
VulkanRenderer::chooseSupportedFormat(const std::vector<VkFormat> &formats,
                                      VkImageTiling tiling,
                                      VkFormatFeatureFlags featureFlags) {
  // loop through candidate formats and check if it supports the feature needed
  for (VkFormat format : formats) {
    // get format properties for format on this device
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(mainDevice.physicalDevice, format,
                                        &properties);
    // check if format supports features for given tiling option
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (properties.linearTilingFeatures & featureFlags) == featureFlags) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (properties.optimalTilingFeatures & featureFlags) ==
                   featureFlags) {
      return format;
    }
  }
  throw std::runtime_error("Failed to find supported format");
}

VkImage VulkanRenderer::createImage(uint32_t width, uint32_t height,
                                    VkFormat format, VkImageTiling tiling,
                                    VkImageUsageFlags useFlags,
                                    VkMemoryPropertyFlags propFlags,
                                    VkDeviceMemory *imageMemory) {
  // CREATE IMAGE
  // image create inof
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D; // type of image (1D, 2D, 3D)
  imageCreateInfo.extent.width = width;         // width of image
  imageCreateInfo.extent.height = height;       // height of image
  imageCreateInfo.extent.depth = 1;             // depth of image (for 3D
                                                // images)
  imageCreateInfo.mipLevels = 1;                // number of mipmap levels
  imageCreateInfo.arrayLayers = 1;              // number of array layers
  imageCreateInfo.format = format;              // format of image data
  imageCreateInfo.tiling =
      tiling; // how image data should be tiled in memory (linear or optimal)
  imageCreateInfo.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED; // initial layout of image
  imageCreateInfo.usage =
      useFlags; // intended usage of image (color attachment, depth stencil
                // attachment, sampled image, etc.)
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT; // number of samples per
                                                   // pixel (for multisampling)
  imageCreateInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // how image will be shared between queues

  // create image
  VkImage image;
  VkResult result = vkCreateImage(mainDevice.logicalDevice, &imageCreateInfo,
                                  nullptr, &image);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image");
  }

  // CREATE MEMORY FOR IMAGE
  // get memory requirements for the image
  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(mainDevice.logicalDevice, image,
                               &memoryRequirements);
  // allocate memory for image
  VkMemoryAllocateInfo memoryAllocInfo = {};
  memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocInfo.allocationSize =
      memoryRequirements.size; // size of memory to allocate (from requirements)
  memoryAllocInfo.memoryTypeIndex = findMemoryTypeIndex(
      mainDevice.physicalDevice, memoryRequirements.memoryTypeBits,
      propFlags); // type of memory to allocate, check
                  // requirements against memory types of
                  // device to find suitable memory type

  result = vkAllocateMemory(mainDevice.logicalDevice, &memoryAllocInfo, nullptr,
                            imageMemory);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate image memory");
  }

  // bind memory to image
  vkBindImageMemory(mainDevice.logicalDevice, image, *imageMemory, 0);

  return image;
}

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format,
                                            VkImageAspectFlags aspectFlags) {
  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.image = image; // image to create view for
  viewCreateInfo.viewType =
      VK_IMAGE_VIEW_TYPE_2D;      // type of image (1D, 2D, 3D, cube)
  viewCreateInfo.format = format; // format of the image data
  viewCreateInfo.components.r =
      VK_COMPONENT_SWIZZLE_IDENTITY; // allows remapping of color to other
                                     // color
  viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  // subresourceRange describes what the image's purpose is and which part
  // of the image to access
  viewCreateInfo.subresourceRange.aspectMask =
      aspectFlags; // which aspect of the image to view (color, depth,
                   // stencil)
  viewCreateInfo.subresourceRange.baseMipLevel =
      0; // start mipmap level to view from
  viewCreateInfo.subresourceRange.levelCount =
      1; // number of mipmap levels to view
  viewCreateInfo.subresourceRange.baseArrayLayer =
      0; // start array layer to view from
  viewCreateInfo.subresourceRange.layerCount =
      1; // number of array layers to view

  // create the image view
  VkImageView imageView;
  VkResult result = vkCreateImageView(mainDevice.logicalDevice, &viewCreateInfo,
                                      nullptr, &imageView);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image view");
  }

  return imageView;
}

VkShaderModule
VulkanRenderer::createShaderModule(const std::vector<char> &code) {
  // shader module creation
  VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.codeSize = code.size();
  shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(
      code.data()); // pointter to code data, need to be uint32_t pointer,
                    // so we cast from char pointer

  VkShaderModule shaderModule;
  VkResult result =
      vkCreateShaderModule(mainDevice.logicalDevice, &shaderModuleCreateInfo,
                           nullptr, &shaderModule);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }

  return shaderModule;
}
