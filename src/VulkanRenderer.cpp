#include "VulkanRenderer.h"
#include "Utilities.h"
#include "VulkanValidation.h"

#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdio>
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
    createGraphicsPipeline();

  } catch (const std::runtime_error &e) {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

void VulkanRenderer::cleanup() {
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

void VulkanRenderer::createRenderPass() {
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

  // attachment reference uses an attachment index to specify which attachment
  // to reference and the layout it will be in during a subpass
  VkAttachmentReference colorAttachmentReference = {};
  colorAttachmentReference.attachment = 0;
  colorAttachmentReference.layout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout of attachment during
                                                // subpass(second transition)

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

  // create info for render pass creation
  VkRenderPassCreateInfo renderPassCreateInfo = {};
  renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreateInfo.attachmentCount =
      1; // number of attachments in render pass
  renderPassCreateInfo.pAttachments =
      &colorAttachment;                  // list of attachments in render pass
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

  // vertex input stage creation info
  VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
  vertexInputCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputCreateInfo.vertexBindingDescriptionCount =
      0; // number of vertex binding descriptions
  vertexInputCreateInfo.pVertexBindingDescriptions =
      nullptr; // list of vertex binding descriptions
  vertexInputCreateInfo.vertexAttributeDescriptionCount =
      0; // number of vertex attribute descriptions
  vertexInputCreateInfo.pVertexAttributeDescriptions =
      nullptr; // list of vertex attribute descriptions

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
      VK_FRONT_FACE_CLOCKWISE; // vertex order for front face
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
      0; // number of descriptor sets to be used in the pipeline
  pipelineLayoutCreateInfo.pSetLayouts =
      nullptr; // list of descriptor set layouts to be used in the pipeline
  pipelineLayoutCreateInfo.pushConstantRangeCount =
      0; // number of push constant ranges to be used in the pipeline
  pipelineLayoutCreateInfo.pPushConstantRanges =
      nullptr; // list of push constant ranges to be used in the pipeline

  // create pipeline layout
  VkResult result = vkCreatePipelineLayout(mainDevice.logicalDevice,
                                           &pipelineLayoutCreateInfo, nullptr,
                                           &pipelineLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }

  // depth and stencil testing

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
      nullptr; // depth and stencil state create info (not using depth or
               // stencil
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

  for (const auto &device : deviceList) {
    if (checkDeviceSuitable(device)) {
      mainDevice.physicalDevice = device;

      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(mainDevice.physicalDevice,
                                    &deviceProperties);

      printf("Selected GPU: %s\n", deviceProperties.deviceName);
      printf("Device Type: %d\n", deviceProperties.deviceType);
      printf("Total Devices Found %d\n", deviceCount);

      printf("API Version: %d.%d.%d\n",
             VK_VERSION_MAJOR(deviceProperties.apiVersion),
             VK_VERSION_MINOR(deviceProperties.apiVersion),
             VK_VERSION_PATCH(deviceProperties.apiVersion));
      break;
    }
  }
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
  // look for mailbox presentation mode, best for triple buffering, low latency
  for (const auto &presentationMode : presentationModes) {
    if (presentationMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return presentationMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRenderer::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
  // if current extent is max, then extent can vary, so we set it to the window
  // size
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

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format,
                                            VkImageAspectFlags aspectFlags) {
  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.image = image; // image to create view for
  viewCreateInfo.viewType =
      VK_IMAGE_VIEW_TYPE_2D;      // type of image (1D, 2D, 3D, cube)
  viewCreateInfo.format = format; // format of the image data
  viewCreateInfo.components.r =
      VK_COMPONENT_SWIZZLE_IDENTITY; // allows remapping of color to other color
  viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  // subresourceRange describes what the image's purpose is and which part of
  // the image to access
  viewCreateInfo.subresourceRange.aspectMask =
      aspectFlags; // which aspect of the image to view (color, depth, stencil)
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
      code.data()); // pointter to code data, need to be uint32_t pointer, so we
                    // cast from char pointer

  VkShaderModule shaderModule;
  VkResult result =
      vkCreateShaderModule(mainDevice.logicalDevice, &shaderModuleCreateInfo,
                           nullptr, &shaderModule);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }

  return shaderModule;
}
