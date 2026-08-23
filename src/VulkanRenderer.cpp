#include "VulkanRenderer.h"
#include "Model.h"
#include "Utilities.h"
#include "VulkanDebug.h"
#include <spdlog/spdlog.h>

#include <chrono>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class CommandThreadPool {
public:
  explicit CommandThreadPool(uint32_t workerCount) {
    workers.reserve(workerCount);
    for (uint32_t i = 0; i < workerCount; ++i)
      workers.emplace_back([this] { workerLoop(); });
  }

  ~CommandThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    workCv.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable())
        worker.join();
    }
  }

  uint32_t size() const { return static_cast<uint32_t>(workers.size()); }

  void dispatch(uint32_t taskCount,
                const std::function<void(uint32_t)> &callback) {
    if (taskCount == 0)
      return;

    {
      std::lock_guard<std::mutex> lock(mutex);
      firstException = nullptr;
      outstandingTasks += taskCount;
      for (uint32_t i = 0; i < taskCount; ++i) {
        tasks.emplace_back([this, callback, i] {
          try {
            callback(i);
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!firstException)
              firstException = std::current_exception();
          }
        });
      }
    }

    workCv.notify_all();
    waitIdle();

    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(mutex);
      failure = firstException;
      firstException = nullptr;
    }
    if (failure)
      std::rethrow_exception(failure);
  }

private:
  void waitIdle() {
    std::unique_lock<std::mutex> lock(mutex);
    doneCv.wait(lock, [this] {
      return tasks.empty() && outstandingTasks == 0;
    });
  }

  void workerLoop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        workCv.wait(lock, [this] { return stopping || !tasks.empty(); });
        if (stopping && tasks.empty())
          return;
        task = std::move(tasks.front());
        tasks.pop_front();
      }

      task();

      {
        std::lock_guard<std::mutex> lock(mutex);
        if (outstandingTasks > 0)
          --outstandingTasks;
        if (tasks.empty() && outstandingTasks == 0)
          doneCv.notify_all();
      }
    }
  }

  std::vector<std::thread> workers;
  std::deque<std::function<void()>> tasks;
  mutable std::mutex mutex;
  std::condition_variable workCv;
  std::condition_variable doneCv;
  uint32_t outstandingTasks = 0;
  bool stopping = false;
  std::exception_ptr firstException;
};

VulkanRenderer::VulkanRenderer() {}

int VulkanRenderer::init(GLFWwindow *newWindow) {
  window = newWindow;

  try {
    // 1. Device
    device.init(window);
    pipelineCache.init(device.getLogicalDevice());

    // 2. Determine formats
    VkFormat swapchainFormat = swapchain.queryImageFormat(device);

    VkFormat colorFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

    shadowDepthFormat = swapchain.chooseSupportedFormat(
        device.getPhysicalDevice(),
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    gBufferDepthFormat = shadowDepthFormat;

    VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;
    litFormat = colorFormat;

    // 3. Render passes
    renderPassManager.createGBufferRenderPass(
        device.getLogicalDevice(), gb0Fmt, gb1Fmt, gb2Fmt, gBufferDepthFormat);
    renderPassManager.createLitRenderPass(device.getLogicalDevice(), litFormat,
                                          gBufferDepthFormat);
    renderPassManager.createSsgiRenderPass(device.getLogicalDevice(), litFormat);
    renderPassManager.createRenderPass(device.getLogicalDevice(),
                                       swapchainFormat, colorFormat,
                                       litFormat);
    renderPassManager.createShadowRenderPass(device.getLogicalDevice(),
                                             shadowDepthFormat);
    renderPassManager.createImGuiRenderPass(device.getLogicalDevice(),
                                            swapchainFormat);

    // 4. Swapchain (images + colorBuffer + command buffers — composite
    //     framebuffers are owned by this class and created below in step 8c)
    swapchain.init(device, window);
    registerSwapchainResources();

    // 5. Shadow resources (directional + point)
    shadowPass.create(device, renderPassManager.getShadowRenderPass(),
                      shadowDepthFormat);

    // 6. Descriptors. The scene UBO is sampled by the async SSAO dispatch on
    // the dedicated compute queue (when one exists), so it needs CONCURRENT
    // graphics+compute sharing — EXCLUSIVE reads across families without an
    // ownership transfer are formally undefined.
    QueueFamilyIndices queueFamilies = device.getQueueFamilies();
    uint32_t sharedUboFamilies[2] = {
        static_cast<uint32_t>(queueFamilies.graphicsFamily),
        static_cast<uint32_t>(queueFamilies.computeFamily)};
    const bool shareUboWithCompute = device.hasDedicatedComputeQueue();
    descriptorManager.init(device.getLogicalDevice(), device.getAllocator(),
                           swapchain.getImageCount(),
                           shareUboWithCompute ? sharedUboFamilies : nullptr,
                           shareUboWithCompute ? 2u : 0u);
    descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
    descriptorManager.updateShadowMapDescriptor(
        device.getLogicalDevice(), shadowPass.csmView(),
        shadowPass.pointCubeView(), shadowPass.csmSampler(),
        shadowPass.pointSampler());

    // 7. Texture manager (samplers only at this point)
    textureManager.init(device, pipelineCache.get());

    // 8. Lit-buffer resources (created BEFORE G-buffer set update so that
    //    recreateGBufferSets has the lit views to bind).
    createLitResources();
    bloomPass.create(device, swapchain.getExtent(), swapchain.getImageCount(),
                     litViews, pipelineCache.get());

    // 8a. SSGI bounce-buffer images + framebuffers. Sampled by lit.frag,
    //     so the G-buffer descriptor set (set 1) gets a 6th binding
    //     pointing at these views.
    ssgiPass.create(device, swapchain.getExtent(), litFormat,
                    renderPassManager.getSsgiRenderPass(),
                    MAX_FRAMES_DRAWS + 1);
    ssaoPass.create(device, swapchain.getExtent(), swapchain.getImageCount(),
                    descriptorManager, pipelineCache.get());

    // 8b. TAA history images. Format matches litFormat so they live in the
    //     same HDR-precision domain.
    createTaaResources();

    // 8c. Composite framebuffers (3 attachments: swap + colorBuffer + history)
    //     Must be created after TAA history images exist and after the
    //     composition render pass is created.
    {
      std::vector<VkImageView> swapViews;
      std::vector<VkImageView> colorViews;
      std::vector<VkImageView> historyViews;
      swapViews.reserve(swapchain.getImageCount());
      colorViews.reserve(swapchain.getImageCount());
      historyViews.reserve(taaHistoryViews.size());
      for (size_t i = 0; i < swapchain.getImageCount(); ++i) {
        swapViews.push_back(swapchain.getSwapImageView(i));
        colorViews.push_back(swapchain.getColorBufferView(i));
      }
      for (const ImageViewHandle &view : taaHistoryViews)
        historyViews.push_back(view.get());
      compositePass.create(device.getLogicalDevice(),
                           renderPassManager.getRenderPass(),
                           swapchain.getExtent(), swapViews, colorViews,
                           historyViews);
    }

    // 9. G-buffer images, framebuffers, descriptor sets (binds lit views too)
    createGBuffer();
    createLitFramebuffers();

    // 9b. TAA descriptor sets (history-prev + depth + colorBuffer-current
    //     + bloom pyramid result).
    //     Needs taaHistory views (from createTaaResources), gBufferDepth
    //     views (from createGBuffer), colorBuffer views (from swapchain init),
    //     and bloom mip 0 views (from BloomPass). Order: must
    //     follow all four.
    {
      std::vector<VkImageView> histViews;
      std::vector<VkImageView> depthViews;
      std::vector<VkImageView> colorViews;
      std::vector<VkImageView> bloomViews;
      histViews.reserve(taaHistoryViews.size());
      depthViews.reserve(gBufferDepthViews.size());
      colorViews.reserve(swapchain.getImageCount());
      bloomViews.reserve(swapchain.getImageCount());
      for (const auto &v : taaHistoryViews)
        histViews.push_back(v.get());
      for (const auto &v : gBufferDepthViews)
        depthViews.push_back(v.get());
      for (size_t i = 0; i < swapchain.getImageCount(); ++i)
        colorViews.push_back(swapchain.getColorBufferView(i));
      bloomViews = bloomPass.mip0Views();
      descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                        depthViews, colorViews, bloomViews,
                                        taaSampler);
    }

    // 10. Pipeline (needs all 5 render passes + descriptor layouts)
    pipeline.createPipelines(
        device.getLogicalDevice(), renderPassManager.getGBufferRenderPass(),
        renderPassManager.getLitRenderPass(), renderPassManager.getRenderPass(),
        renderPassManager.getShadowRenderPass(),
        renderPassManager.getSsgiRenderPass(), swapchain.getExtent(),
        pipelineCache.get(), descriptorManager);
    gpuDrivenGBufferPass.create(device, modelManager, descriptorManager,
                                renderPassManager.getGBufferRenderPass(),
                                swapchain.getExtent(), gBufferDepthViews,
                                pipelineCache.get());
    materialProbePass.create(device, MAX_FRAMES_DRAWS, descriptorManager,
                             pipelineCache.get());

    // 10. IBL resources (requires textureManager and descriptorManager)
    initIBL();

    // 10b. Auto-exposure compute resources. Must come after createLitResources()
    //      (binds litViews) and after swapchain init (knows image count).
    autoExposurePass.create(device, swapchain.getExtent(), litViews,
                            litSampler, pipelineCache.get());

    // 11. Synchronization
    createSynchronization();
    createAsyncFrameCommandBuffers();
    createThreadedCommandResources();

    // 12. Performance metrics
    QueueFamilyIndices qi = device.getQueueFamilies();
    metrics.init(device.getLogicalDevice(), device.getPhysicalDevice(),
                 static_cast<uint32_t>(qi.graphicsFamily), MAX_FRAMES_DRAWS);

    // 13. Scene UBO defaults
    rebuildProjection();

    sceneUbo.view =
        glm::lookAt(glm::vec3(10.0f, 0.0f, 20.0f), glm::vec3(0.0f, 0.0f, -2.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    sceneUbo.invView = glm::inverse(sceneUbo.view);
    sceneUbo.invProj = glm::inverse(sceneUbo.projection);

    sceneUbo.cameraPosition = glm::vec4(10.0f, 0.0f, 20.0f, 1.0f);

    // Warm midday sun coming through the atrium roof openings
    sceneUbo.directionalLight.direction =
        glm::vec4(glm::normalize(glm::vec3(-0.3f, -1.0f, 0.2f)), 0.0f);
    // Intensity 2.5 (was 5.5) — at 5.5 the floor pixels under direct sun
    // overshoot AgX's mid-tone band and the tile pattern crushes into a
    // uniform bright wash. 2.5 lands the lit floor in AgX's preserved-
    // detail range so authored albedo+normal variation actually survives
    // the tonemap. This is a stopgap until the histogram auto-exposure
    // (N1 in mds/sponza_visual_diagnosis.md) can drive exposure
    // automatically; once that lands, sun intensity becomes a free knob.
    sceneUbo.directionalLight.colorIntensity =
        glm::vec4(1.0f, 0.93f, 0.78f, 2.5f);

    // Warm amber torchlight on the arcade pillars (arch height ~4.5m, x=±8.5m)
    sceneUbo.pointLights[0].position = glm::vec4(8.5f, 4.5f, 2.0f, 1.0f);
    sceneUbo.pointLights[0].colorIntensity =
        glm::vec4(1.0f, 0.70f, 0.35f, 6.0f);
    sceneUbo.pointLights[1].position = glm::vec4(-8.5f, 4.5f, -2.0f, 1.0f);
    sceneUbo.pointLights[1].colorIntensity =
        glm::vec4(1.0f, 0.68f, 0.32f, 6.0f);

    sceneUbo.spotLights[0].position = glm::vec4(0.0f, 5.0f, 6.0f, 1.0f);
    sceneUbo.spotLights[0].direction =
        glm::vec4(glm::normalize(glm::vec3(0.0f, -0.75f, -1.0f)), 0.0f);
    sceneUbo.spotLights[0].colorIntensity = glm::vec4(1.0f, 0.88f, 0.65f, 1.5f);
    sceneUbo.spotLights[0].cutoffAngles =
        glm::vec4(glm::cos(glm::radians(12.5f)), glm::cos(glm::radians(20.0f)),
                  0.0f, 0.0f);

    // shadowParams.y/.z/.w piggyback post/SSGI tunables since they were unused.
    // .x stays as the point-shadow far plane; .w toggles SSR.
    sceneUbo.shadowParams = glm::vec4(imguiPointShadowFar, imguiSsgiIntensity,
                                      imguiSharpness,
                                      imguiSsrEnabled ? 1.0f : 0.0f);
    sceneUbo.visualToggles =
        glm::vec4(imguiEdgeAA, imguiAlphaDither, imguiShadowSoftness,
                  imguiContactGrounding);
    sceneUbo.fogParams =
        glm::vec4(imguiFogDensity, 0.25f, imguiFogClamp,
                  static_cast<float>(std::clamp(imguiSsgiSamples, 4, 12)));
    sceneUbo.lightCounts =
        glm::ivec4(1, 0, 0x3f, 0); // z = lighting-isolation bit mask

    shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                        imguiDrawDistance);
    shadowPass.updatePointShadowMatrices(sceneUbo);

    // 14. Default albedo texture
    textureManager.loadTexture("plain.png", device, descriptorManager);

    // 15. ImGui overlay
    imguiLayer.init(window, device, renderPassManager.getImGuiRenderPass(),
                    static_cast<uint32_t>(swapchain.getImageCount()));
    imguiLayer.createFramebuffers(device.getLogicalDevice(),
                                  renderPassManager.getImGuiRenderPass(),
                                  swapchain.getExtent(), swapchain.getImages());

  } catch (const std::runtime_error &e) {
    spdlog::critical("Renderer initialization failed: {}", e.what());
    return EXIT_FAILURE;
  }
  return 0;
}

int VulkanRenderer::createMeshModel(const std::string &modelFile) {
  int modelId = modelManager.loadModel(modelFile, device, textureManager,
                                       descriptorManager);
  gpuDrivenGBufferPass.registerModelGeometry(modelId, modelManager);
  return modelId;
}

void VulkanRenderer::discoverScenes(const std::string &directory) {
  availableScenes = SceneIO::discover(directory);
  availableSceneNames.clear();
  availableSceneNames.reserve(availableScenes.size());
  for (const SceneDescription &scene : availableScenes)
    availableSceneNames.push_back(scene.name);
  spdlog::info("Discovered {} scene file(s) in {}", availableScenes.size(),
               directory);
}

void VulkanRenderer::loadScene(const SceneDescription &desc) {
  // Rare, user-triggered operation: a full idle wait is the simple and safe
  // way to retire every in-flight frame before instance buffers and the
  // scene graph are torn down.
  vkDeviceWaitIdle(device.getLogicalDevice());

  rootNode.clearChildren();
  instancedDrawables.clear(); // AllocatedBuffer RAII frees instance buffers

  for (const SceneModelEntry &entry : desc.models) {
    try {
      int modelId = createMeshModel(entry.path);
      auto node = std::make_unique<SceneNode>();
      node->setModelId(modelId);
      node->setTransform(entry.transform);
      node->setAnimation(entry.animation);
      rootNode.addChild(std::move(node));
    } catch (const std::exception &e) {
      spdlog::error("Scene '{}': failed to load model '{}': {}", desc.name,
                    entry.path, e.what());
    }
  }

  for (const SceneRingEntry &ring : desc.rings) {
    if (ring.count <= 0)
      continue;
    try {
      int modelId = createMeshModel(ring.path);
      std::vector<glm::mat4> transforms;
      transforms.reserve(static_cast<size_t>(ring.count));
      for (int i = 0; i < ring.count; i++) {
        float angle = glm::two_pi<float>() * static_cast<float>(i) /
                      static_cast<float>(ring.count);
        glm::vec3 pos(ring.radius * std::cos(angle), ring.height,
                      ring.radius * std::sin(angle));
        glm::mat4 t = glm::translate(glm::mat4(1.0f), pos);
        t = glm::rotate(t, angle, glm::vec3(0.0f, 1.0f, 0.0f));
        t = glm::scale(t, glm::vec3(ring.scale));
        transforms.push_back(t);
      }
      addInstancedModel(modelId, transforms);
    } catch (const std::exception &e) {
      spdlog::error("Scene '{}': failed to load ring model '{}': {}", desc.name,
                    ring.path, e.what());
    }
  }

  rootNode.update(glm::mat4(1.0f), 0.0f);

  if (!desc.environmentHdr.empty() &&
      desc.environmentHdr != activeEnvironmentHdr) {
    spdlog::warn("Scene '{}' requests environment '{}', but runtime HDRI "
                 "switching is not implemented yet; keeping the current sky.",
                 desc.name, desc.environmentHdr);
  }

  if (cameraPresetCallback)
    cameraPresetCallback(desc.camera.position, desc.camera.yaw,
                         desc.camera.pitch, desc.camera.speed);

  // The new scene shares no image history with the old one. The HZB pyramid
  // must be invalidated too — occlusion-culling the new scene against the
  // old scene's depth pops meshes for the first frames after a switch.
  // (The SSGI history ring keeps a few frames of the old scene's bounce
  // light; that ghost fades within ~10 frames of temporal blend and is
  // accepted rather than adding a clear path.)
  taaHistoryValid = false;
  gpuDrivenGBufferPass.invalidateHzb();

  spdlog::info("Loaded scene '{}' ({} model(s), {} instanced ring(s))",
               desc.name, desc.models.size(), desc.rings.size());
}

bool VulkanRenderer::loadSceneAt(int index) {
  if (index < 0 || index >= static_cast<int>(availableScenes.size())) {
    spdlog::error("loadSceneAt: index {} out of range ({} scenes)", index,
                  availableScenes.size());
    return false;
  }
  loadScene(availableScenes[index]);
  activeSceneIndex = index;
  return true;
}

void VulkanRenderer::requestSceneLoad(int index) {
  pendingSceneIndex = index;
}

void VulkanRenderer::updateScene(float timeSeconds) {
  rootNode.update(glm::mat4(1.0f), timeSeconds);
}

SceneNode &VulkanRenderer::getRootNode() { return rootNode; }

void VulkanRenderer::updateCameraView(const glm::mat4 &viewMatrix,
                                      const glm::vec3 &cameraPosition) {
  if (taaHasLastCamera) {
    float posDelta = glm::length(cameraPosition - taaLastCameraPos);
    float viewDelta = 0.0f;
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        viewDelta = std::max(
            viewDelta, std::abs(viewMatrix[c][r] - taaLastView[c][r]));
      }
    }
    cameraMovedThisFrame = posDelta > 0.001f || viewDelta > 0.0001f;
  } else {
    cameraMovedThisFrame = true;
    taaHasLastCamera = true;
  }
  taaLastCameraPos = cameraPosition;
  taaLastView = viewMatrix;
  sceneUbo.view = viewMatrix;
  sceneUbo.cameraPosition = glm::vec4(cameraPosition, 1.0f);
  sceneUbo.invView = glm::inverse(viewMatrix);
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
}

void VulkanRenderer::notifyResize() { framebufferResized = true; }

void VulkanRenderer::updateMaterialProbePixel() {
  double cursorX = 0.0;
  double cursorY = 0.0;
  glfwGetCursorPos(window, &cursorX, &cursorY);

  int fbWidth = 0;
  int fbHeight = 0;
  glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
  VkExtent2D extent = swapchain.getExtent();
  if (fbWidth <= 0 || fbHeight <= 0 || extent.width == 0 ||
      extent.height == 0) {
    materialProbePixel = glm::uvec2(extent.width, extent.height);
    return;
  }

  const double scaledX = cursorX * double(extent.width) / double(fbWidth);
  const double scaledY = cursorY * double(extent.height) / double(fbHeight);
  if (scaledX < 0.0 || scaledY < 0.0 || scaledX >= double(extent.width) ||
      scaledY >= double(extent.height)) {
    materialProbePixel = glm::uvec2(extent.width, extent.height);
    return;
  }

  materialProbePixel =
      glm::uvec2(static_cast<uint32_t>(scaledX), static_cast<uint32_t>(scaledY));
}

// Draw

void VulkanRenderer::draw() {
  VkDevice logicalDevice = device.getLogicalDevice();

  // Apply a UI-requested scene switch before any of this frame's work
  // starts. loadSceneAt() waits for device idle, so nothing recorded or
  // submitted this frame can reference torn-down scene resources.
  if (pendingSceneIndex >= 0) {
    int requested = pendingSceneIndex;
    pendingSceneIndex = -1;
    loadSceneAt(requested);
  }

  metrics.beginFrame();

  using CpuClock = std::chrono::high_resolution_clock;
  auto elapsedMs = [](CpuClock::time_point start) {
    return std::chrono::duration<double, std::milli>(CpuClock::now() - start)
        .count();
  };

  auto phaseStart = CpuClock::now();
  vkWaitForFences(logicalDevice, 1, &drawFences[currentFrame], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::WaitFence,
                         elapsedMs(phaseStart));
  // This is the expected frame-in-flight throttle: if the CPU finishes
  // recording faster than the GPU can consume work, it waits here before
  // reusing this frame slot's fence, acquire semaphore, query range, and
  // exposure result buffer. It is idle time, not active CPU renderer cost.
  metrics.collectGpuResults(logicalDevice, currentFrame);
  materialProbeResult = materialProbePass.read(currentFrame);

  // Check resize BEFORE acquiring — a successful acquire signals
  // imageAvailable, and returning early without consuming it would leave it
  // signaled on the next frame.
  if (framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  }

  uint32_t imageIndex;
  phaseStart = CpuClock::now();
  VkResult acquireResult = vkAcquireNextImageKHR(
      logicalDevice, swapchain.getSwapchain(),
      std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame],
      VK_NULL_HANDLE, &imageIndex);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Acquire,
                         elapsedMs(phaseStart));

  if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
    // Failed acquire → semaphore NOT signaled per spec → safe to recreate
    recreateSwapChain();
    return;
  } else if (acquireResult != VK_SUCCESS &&
             acquireResult != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  phaseStart = CpuClock::now();
  VkFence imageFence = imagesInFlight[imageIndex];
  if (imageFence != VK_NULL_HANDLE) {
    bool fenceStillOwnsImage = false;
    for (size_t frame = 0; frame < drawFences.size(); ++frame) {
      if (drawFences[frame] == imageFence) {
        fenceStillOwnsImage =
            frame < frameImageInFlight.size() &&
            frameImageInFlight[frame] == imageIndex;
        break;
      }
    }

    if (fenceStillOwnsImage) {
      vkWaitForFences(logicalDevice, 1, &imageFence, VK_TRUE, UINT64_MAX);
    } else {
      imagesInFlight[imageIndex] = VK_NULL_HANDLE;
    }
  }
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::ImageFence,
                         elapsedMs(phaseStart));

  const uint32_t invalidImageIndex = std::numeric_limits<uint32_t>::max();
  if (currentFrame < static_cast<int>(frameImageInFlight.size())) {
    uint32_t previousImage = frameImageInFlight[currentFrame];
    if (previousImage != invalidImageIndex &&
        previousImage < imagesInFlight.size() &&
        imagesInFlight[previousImage] == drawFences[currentFrame]) {
      imagesInFlight[previousImage] = VK_NULL_HANDLE;
    }
    frameImageInFlight[currentFrame] = imageIndex;
  }
  imagesInFlight[imageIndex] = drawFences[currentFrame];
  vkResetFences(logicalDevice, 1, &drawFences[currentFrame]);

  phaseStart = CpuClock::now();
  // Day/night cycle: advance sim-hour at imguiDayNightSpeed sim-hours per
  // real-second, then place the sun on a great-circle arc through the
  // zenith. Sunrise at hour 6 (east horizon), zenith at hour 12, sunset at
  // hour 18 (west horizon); below the horizon outside [6, 18].
  if (imguiDayNightEnable) {
    static auto lastTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;
    imguiDayNightHour += dt * imguiDayNightSpeed;
    imguiDayNightHour = std::fmod(imguiDayNightHour, 24.0f);
    if (imguiDayNightHour < 0.0f) imguiDayNightHour += 24.0f;

    float t     = (imguiDayNightHour - 6.0f) / 12.0f; // 0 at dawn, 1 at dusk
    float angle = t * 3.1415927f;                     // 0..π over daylight
    // Slight tilt off the X axis so shadows aren't perfectly E-W parallel.
    glm::vec3 east = glm::normalize(glm::vec3(1.0f, 0.0f, 0.2f));
    glm::vec3 up   = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 sunPos = east * std::cos(angle) + up * std::sin(angle);
    sceneUbo.directionalLight.direction = glm::vec4(-sunPos, 0.0f);

    // Intensity follows elevation: max at zenith, ambient floor at night.
    // Warm tint kicks in only near the horizon, not while underground.
    float elevation = sunPos.y;
    float dayFactor = glm::smoothstep(-0.05f, 0.35f, elevation);
    float horizon   = glm::clamp(1.0f - std::abs(elevation) * 3.0f, 0.0f, 1.0f) *
                      (elevation > 0.0f ? 1.0f : 0.0f);
    glm::vec3 dayColor(1.0f, 0.93f, 0.78f);
    glm::vec3 duskColor(1.0f, 0.55f, 0.30f);
    glm::vec3 color = glm::mix(dayColor, duskColor, horizon);
    // Peak 2.5 (was 5.5) — see comment in createSceneUBO. Stopgap until
    // histogram auto-exposure replaces the manual EV slider.
    sceneUbo.directionalLight.colorIntensity =
        glm::vec4(color, 2.5f * dayFactor + 0.05f);

    // IBL/sky baked from a daytime cubemap — attenuate it with the same
    // day factor so glossy surfaces don't keep reflecting baked daylight
    // through the night. User's manual slider still scales on top.
    sceneUbo.qualityToggles2.w =
        imguiIblIntensity * (0.08f + 0.92f * dayFactor);
  } else {
    // Day/night cycle off: manual slider drives the ambient term directly.
    sceneUbo.qualityToggles2.w = imguiIblIntensity;
  }

  sceneUbo.qualityToggles2.y =
      imguiUseGeomNormalOnly ? 0.0f : imguiNormalStrength;
  sceneUbo.qualityToggles2.x =
      autoExposurePass.updateExposureScale(currentFrame, autoExpEnabled,
                                           imguiExposureEV);
  sceneUbo.qualityToggles.x  = imguiIblRoughnessFloor;
  sceneUbo.qualityToggles2.z = imguiMinSurfaceRoughness;
  sceneUbo.qualityToggles.y  = imguiSkyOcclusionFloor;
  sceneUbo.qualityToggles.z  = imguiSpecAAVariance;
  sceneUbo.qualityToggles.w  = imguiSpecAAThreshold;
  int lightingMask = 0;
  if (imguiEnableSunDirect)   lightingMask |= 1 << 0;
  if (imguiEnablePointLights) lightingMask |= 1 << 1;
  if (imguiEnableSpotLights)  lightingMask |= 1 << 2;
  if (imguiEnableIblAmbient)  lightingMask |= 1 << 3;
  if (imguiEnableSsgiBounce)  lightingMask |= 1 << 4;
  if (imguiEnableBloom)       lightingMask |= 1 << 5;
  sceneUbo.lightCounts.z = lightingMask;
  sceneUbo.fogParams.w =
      static_cast<float>(std::clamp(imguiSsgiSamples, 4, 12));
  // shadowParams.y = active SSGI intensity, .z = sharpening strength.
  // If SSGI bounce is disabled in lighting isolation, keep the pass cheap by
  // making ssgi.frag return immediately instead of running the sample loop.
  sceneUbo.shadowParams.y    =
      imguiEnableSsgiBounce ? imguiSsgiIntensity : 0.0f;
  sceneUbo.shadowParams.z    = imguiSharpness;
  sceneUbo.shadowParams.w    = imguiSsrEnabled ? 1.0f : 0.0f;
  sceneUbo.visualToggles =
      glm::vec4(imguiEdgeAA, imguiAlphaDither, imguiShadowSoftness,
                imguiContactGrounding);

  // --- TAA per-frame state ---
  // Halton(2,3) sub-pixel jitter for free supersampling AA + temporal noise
  // washing. We sequence over 8 samples and wrap. Convert pixel offsets in
  // [-0.5, 0.5] to NDC offsets by *2/viewport, since clip.xy is in [-w, w]
  // and after divide we want a *2/viewport NDC shift to land on the
  // intended sub-pixel position.
  auto halton = [](uint32_t i, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (i > 0) { f /= float(base); r += f * float(i % base); i /= base; }
    return r;
  };
  uint32_t haltonIdx = (taaFrameCounter % 8) + 1; // skip i=0 → (0,0)
  glm::vec2 jitterPx = glm::vec2(halton(haltonIdx, 2) - 0.5f,
                                 halton(haltonIdx, 3) - 0.5f);
  VkExtent2D ext = swapchain.getExtent();
  glm::vec2 jitterNDC = glm::vec2(jitterPx.x * 2.0f / float(ext.width),
                                  jitterPx.y * 2.0f / float(ext.height));
  // prevViewProj must hold the *jittered* VP of the prev frame so that
  // multiplying a world position by it yields the prev frame's *jittered*
  // NDC — the location at which that surface point was actually rasterized
  // in the history image. Sampling history at the un-jittered prev NDC was
  // off by half a pixel each frame, which bilinear-blurred the image over
  // many frames (a textbook TAA blur bug).
  sceneUbo.prevViewProj = taaPrevViewProj;
  // Reset projection to the un-jittered baseline BEFORE applying this
  // frame's Halton offset. The previous implementation mutated
  // sceneUbo.projection in-place every frame, which accumulated the
  // per-frame offsets into a random-walk drift over time — the relation
  // between Halton index and actual sub-pixel position broke, so TAA
  // could never converge to clean supersampled edges (the symptom the
  // user reported: "textures slightly move all the time", and severely
  // jagged edges even on a static camera).
  sceneUbo.projection = taaBaseProjection;
  const bool taaStable = !imguiResponsiveTaa || !cameraMovedThisFrame;
  const bool taaEnable = imguiTaaEnabled && taaStable;
  if (taaEnable) {
    // Modify column 2 (the z coupling) so the offset is a constant screen-
    // space delta independent of vertex depth: clip.x_new = clip.x +
    // jitterNDC.x * w. After perspective divide this becomes a constant
    // +jitterNDC NDC offset on every fragment regardless of depth.
    sceneUbo.projection[2][0] -= jitterNDC.x;
    sceneUbo.projection[2][1] -= jitterNDC.y;
  }
  // invProj must invert whichever projection was just installed.
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
  // Save THIS frame's jittered VP for next frame's reprojection. Must come
  // AFTER the jitter mutation above.
  taaPrevViewProj = sceneUbo.projection * sceneUbo.view;
  sceneUbo.taaParams = glm::vec4(taaEnable ? jitterNDC.x : 0.0f,
                                 taaEnable ? jitterNDC.y : 0.0f,
                                 taaEnable ? 1.0f : 0.0f,
                                 (taaHistoryValid && taaStable) ? 1.0f : 0.0f);
  sceneUbo.viewportSize = glm::vec4(float(ext.width), float(ext.height),
                                    1.0f / float(ext.width),
                                    1.0f / float(ext.height));
  // taaHistoryValid is the input to THIS frame's TAA blend (so the first
  // frame after init/resize must read it as false). Promote it to true for
  // the NEXT frame now that we're about to render content into the history.
  taaHistoryValid = taaEnable;

  shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                      imguiDrawDistance);
  shadowPass.updatePointShadowMatrices(sceneUbo);
  updateMaterialProbePixel();
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Update,
                         elapsedMs(phaseStart));

  metrics.setActiveGpuQueryFrame(currentFrame);
  phaseStart = CpuClock::now();
  recordGBufferCommands(imageIndex);
  recordSsaoComputeCommands(ssaoCommandBuffers[imageIndex], imageIndex);
  recordShadowCommands(shadowCommandBuffers[imageIndex]);
  recordPostCommands(postCommandBuffers[imageIndex], imageIndex);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Record,
                         elapsedMs(phaseStart));

  phaseStart = CpuClock::now();
  descriptorManager.updateUniformBuffer(device.getAllocator(), imageIndex,
                                        &sceneUbo, sizeof(SceneUniformBuffer));
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Upload,
                         elapsedMs(phaseStart));

  phaseStart = CpuClock::now();
  VkSemaphore timeline = asyncComputeTimeline[currentFrame];
  const uint64_t gbufferReadyValue =
      asyncComputeTimelineValue[currentFrame] + 1;
  const uint64_t ssaoReadyValue = asyncComputeTimelineValue[currentFrame] + 2;
  asyncComputeTimelineValue[currentFrame] = ssaoReadyValue;

  VkCommandBuffer gbufferCmd = swapchain.getCommandBuffer(imageIndex);
  VkCommandBuffer ssaoCmd = ssaoCommandBuffers[imageIndex];
  VkCommandBuffer shadowCmd = shadowCommandBuffers[imageIndex];
  VkCommandBuffer postCmd = postCommandBuffers[imageIndex];

  uint64_t zeroValue = 0;
  VkPipelineStageFlags imageWaitStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkTimelineSemaphoreSubmitInfo gbufferTimelineInfo = {};
  gbufferTimelineInfo.sType =
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  gbufferTimelineInfo.waitSemaphoreValueCount = 1;
  gbufferTimelineInfo.pWaitSemaphoreValues = &zeroValue; // binary wait
  gbufferTimelineInfo.signalSemaphoreValueCount = 1;
  gbufferTimelineInfo.pSignalSemaphoreValues = &gbufferReadyValue;
  VkSubmitInfo gbufferSubmit = {};
  gbufferSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  gbufferSubmit.pNext = &gbufferTimelineInfo;
  gbufferSubmit.waitSemaphoreCount = 1;
  gbufferSubmit.pWaitSemaphores = &imageAvailable[currentFrame];
  gbufferSubmit.pWaitDstStageMask = &imageWaitStage;
  gbufferSubmit.commandBufferCount = 1;
  gbufferSubmit.pCommandBuffers = &gbufferCmd;
  gbufferSubmit.signalSemaphoreCount = 1;
  gbufferSubmit.pSignalSemaphores = &timeline;
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &gbufferSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit G-buffer command buffer");
  }

  VkPipelineStageFlags computeWaitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfo ssaoTimelineInfo = {};
  ssaoTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  ssaoTimelineInfo.waitSemaphoreValueCount = 1;
  ssaoTimelineInfo.pWaitSemaphoreValues = &gbufferReadyValue;
  ssaoTimelineInfo.signalSemaphoreValueCount = 1;
  ssaoTimelineInfo.pSignalSemaphoreValues = &ssaoReadyValue;
  VkSubmitInfo ssaoSubmit = {};
  ssaoSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  ssaoSubmit.pNext = &ssaoTimelineInfo;
  ssaoSubmit.waitSemaphoreCount = 1;
  ssaoSubmit.pWaitSemaphores = &timeline;
  ssaoSubmit.pWaitDstStageMask = &computeWaitStage;
  ssaoSubmit.commandBufferCount = 1;
  ssaoSubmit.pCommandBuffers = &ssaoCmd;
  ssaoSubmit.signalSemaphoreCount = 1;
  ssaoSubmit.pSignalSemaphores = &timeline;
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getComputeQueue(), 1, &ssaoSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit SSAO compute command buffer");
  }

  VkSubmitInfo shadowSubmit = {};
  shadowSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  shadowSubmit.commandBufferCount = 1;
  shadowSubmit.pCommandBuffers = &shadowCmd;
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &shadowSubmit,
                    VK_NULL_HANDLE) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit shadow command buffer");
  }

  uint64_t renderFinishedSignalValue = 0; // binary signal, ignored
  VkPipelineStageFlags postWaitStage =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfo postTimelineInfo = {};
  postTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  postTimelineInfo.waitSemaphoreValueCount = 1;
  postTimelineInfo.pWaitSemaphoreValues = &ssaoReadyValue;
  postTimelineInfo.signalSemaphoreValueCount = 1;
  postTimelineInfo.pSignalSemaphoreValues = &renderFinishedSignalValue;
  VkSubmitInfo postSubmit = {};
  postSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  postSubmit.pNext = &postTimelineInfo;
  postSubmit.waitSemaphoreCount = 1;
  postSubmit.pWaitSemaphores = &timeline;
  postSubmit.pWaitDstStageMask = &postWaitStage;
  postSubmit.commandBufferCount = 1;
  postSubmit.pCommandBuffers = &postCmd;
  postSubmit.signalSemaphoreCount = 1;
  // renderFinished is indexed by acquired swap-image, not by frame-in-flight,
  // so the presentation engine's wait binds to the right semaphore.
  postSubmit.pSignalSemaphores = &renderFinished[imageIndex];
  metrics.recordQueueSubmit();
  if (vkQueueSubmit(device.getGraphicsQueue(), 1, &postSubmit,
                    drawFences[currentFrame]) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit post command buffer");
  }
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Submit,
                         elapsedMs(phaseStart));
  metrics.markGpuQueriesSubmitted(currentFrame);

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[imageIndex];
  presentInfo.swapchainCount = 1;
  VkSwapchainKHR sc = swapchain.getSwapchain();
  presentInfo.pSwapchains = &sc;
  presentInfo.pImageIndices = &imageIndex;

  phaseStart = CpuClock::now();
  VkResult result =
      vkQueuePresentKHR(device.getPresentationQueue(), &presentInfo);
  metrics.recordCpuPhase(PerformanceMetrics::CpuPhase::Present,
                         elapsedMs(phaseStart));
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
  currentFrame = (currentFrame + 1) % MAX_FRAMES_DRAWS;
  // Advance TAA frame counter AFTER this frame finishes recording so the
  // Halton jitter index used above matches the history ring slots that
  // recordCommands picked for framebuffers / descriptor sets.
  taaFrameCounter++;
  metrics.endFrame();
}

// Swapchain recreation

void VulkanRenderer::recreateSwapChain() {
  vkDeviceWaitIdle(device.getLogicalDevice());
  renderResources.clear();
  cleanupAsyncFrameCommandBuffers();
  swapchain.recreate(device, window);
  registerSwapchainResources();

  // Present-mode toggles commonly change minImageCount; every per-image
  // descriptor resource must track the new count or acquired indices run
  // past the init-time arrays.
  {
    QueueFamilyIndices queueFamilies = device.getQueueFamilies();
    uint32_t sharedUboFamilies[2] = {
        static_cast<uint32_t>(queueFamilies.graphicsFamily),
        static_cast<uint32_t>(queueFamilies.computeFamily)};
    const bool shareUbo = device.hasDedicatedComputeQueue();
    if (descriptorManager.recreatePerImageResources(
            device.getLogicalDevice(), device.getAllocator(),
            swapchain.getImageCount(), shareUbo ? sharedUboFamilies : nullptr,
            shareUbo ? 2u : 0u)) {
      descriptorManager.updateShadowMapDescriptor(
          device.getLogicalDevice(), shadowPass.csmView(),
          shadowPass.pointCubeView(), shadowPass.csmSampler(),
          shadowPass.pointSampler());
      descriptorManager.updateIblDescriptors(
          device.getLogicalDevice(),
          textureManager.getImageView(irradianceImageIndex), iblSampler,
          textureManager.getImageView(prefilteredEnvImageIndex),
          textureManager.getImageView(brdfLutImageIndex),
          textureManager.getImageView(ssaoNoiseImageIndex), ssaoNoiseSampler,
          textureManager.getImageView(iblSkyboxImageIndex));
    }
  }

  imguiLayer.createFramebuffers(device.getLogicalDevice(),
                                renderPassManager.getImGuiRenderPass(),
                                swapchain.getExtent(), swapchain.getImages());

  autoExposurePass.cleanup();
  compositePass.cleanup();
  gpuDrivenGBufferPass.cleanup();
  cleanupLitFramebuffers();
  cleanupGBuffer();
  ssaoPass.cleanup();
  bloomPass.cleanup();
  ssgiPass.cleanup();
  cleanupLitResources();
  cleanupTaaResources();
  createLitResources();
  bloomPass.create(device, swapchain.getExtent(), swapchain.getImageCount(),
                   litViews, pipelineCache.get());
  ssgiPass.create(device, swapchain.getExtent(), litFormat,
                  renderPassManager.getSsgiRenderPass(),
                  MAX_FRAMES_DRAWS + 1);
  ssaoPass.create(device, swapchain.getExtent(), swapchain.getImageCount(),
                  descriptorManager, pipelineCache.get());
  createTaaResources();
  {
    std::vector<VkImageView> swapViews;
    std::vector<VkImageView> colorViews;
    std::vector<VkImageView> historyViews;
    swapViews.reserve(swapchain.getImageCount());
    colorViews.reserve(swapchain.getImageCount());
    historyViews.reserve(taaHistoryViews.size());
    for (size_t i = 0; i < swapchain.getImageCount(); ++i) {
      swapViews.push_back(swapchain.getSwapImageView(i));
      colorViews.push_back(swapchain.getColorBufferView(i));
    }
    for (const ImageViewHandle &view : taaHistoryViews)
      historyViews.push_back(view.get());
    compositePass.create(device.getLogicalDevice(),
                         renderPassManager.getRenderPass(),
                         swapchain.getExtent(), swapViews, colorViews,
                         historyViews);
  }
  createGBuffer();
  createLitFramebuffers();
  gpuDrivenGBufferPass.create(device, modelManager, descriptorManager,
                              renderPassManager.getGBufferRenderPass(),
                              swapchain.getExtent(), gBufferDepthViews,
                              pipelineCache.get());
  // Recreate auto-exposure AFTER lit (descriptors reference litViews).
  autoExposurePass.create(device, swapchain.getExtent(), litViews, litSampler,
                          pipelineCache.get());

  imagesInFlight.assign(swapchain.getImageCount(), VK_NULL_HANDLE);
  frameImageInFlight.assign(MAX_FRAMES_DRAWS,
                            std::numeric_limits<uint32_t>::max());

  // renderFinished is sized by swap-image count; if the new swapchain has
  // a different count, resize and (re)create the missing semaphores.
  // Destroying and re-creating all of them is cheap and avoids stale
  // signaled state from before the resize.
  {
    VkDevice dev = device.getLogicalDevice();
    for (VkSemaphore s : renderFinished)
      vkDestroySemaphore(dev, s, nullptr);
    renderFinished.assign(swapchain.getImageCount(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < renderFinished.size(); ++i) {
      if (vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
          VK_SUCCESS)
        throw std::runtime_error("Failed to recreate renderFinished semaphore");
    }
  }

  descriptorManager.recreateInputSets(device.getLogicalDevice(), swapchain);
  {
    std::vector<VkImageView> histViews;
    std::vector<VkImageView> depthViews;
    std::vector<VkImageView> colorViews;
    std::vector<VkImageView> bloomViews;
    histViews.reserve(taaHistoryViews.size());
    depthViews.reserve(gBufferDepthViews.size());
    colorViews.reserve(swapchain.getImageCount());
    bloomViews.reserve(swapchain.getImageCount());
    for (const auto &v : taaHistoryViews)
      histViews.push_back(v.get());
    for (const auto &v : gBufferDepthViews)
      depthViews.push_back(v.get());
    for (size_t i = 0; i < swapchain.getImageCount(); ++i)
      colorViews.push_back(swapchain.getColorBufferView(i));
    bloomViews = bloomPass.mip0Views();
    descriptorManager.recreateTaaSets(device.getLogicalDevice(), histViews,
                                      depthViews, colorViews, bloomViews,
                                      taaSampler);
  }

  createAsyncFrameCommandBuffers();
  rebuildProjection();
}

void VulkanRenderer::registerSwapchainResources() {
  renderResources.clear();
  const size_t count = swapchain.getImageCount();
  renderResources.resizeFrames(count);

  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  const VkSharingMode swapSharing =
      queues.graphicsFamily != queues.presentationFamily
          ? VK_SHARING_MODE_CONCURRENT
          : VK_SHARING_MODE_EXCLUSIVE;

  for (size_t i = 0; i < count; ++i) {
    // Freshly created swapchain images are in UNDEFINED, not PRESENT_SRC —
    // they have never been presented. Registering PRESENT_SRC would make a
    // future RenderResources::transition() emit an invalid oldLayout.
    renderResources.registerImage(
        {"swapchain[" + std::to_string(i) + "]", swapchain.getSwapImage(i),
         swapchain.getImageFormat(), VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, swapSharing, graphicsFamily});
    renderResources.registerImage(
        {"compositeColor[" + std::to_string(i) + "]",
         swapchain.getColorBufferImage(i), swapchain.getColorBufferFormat(),
         VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_SHARING_MODE_EXCLUSIVE, graphicsFamily});

    FrameRenderTargets &targets = renderResources.frame(i);
    targets.swapchainImage = swapchain.getSwapImage(i);
    targets.colorBuffer = swapchain.getColorBufferImage(i);
  }
}

void VulkanRenderer::noteGBufferFinalLayouts(uint32_t imageIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  constexpr VkPipelineStageFlags colorStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  constexpr VkAccessFlags colorAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  renderResources.noteLayout(targets.gBuffer0,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBuffer1,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBuffer2,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             colorStage, colorAccess);
  renderResources.noteLayout(targets.gBufferDepth,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

void VulkanRenderer::noteLitFinalLayout(uint32_t imageIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  renderResources.noteLayout(targets.lit,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

void VulkanRenderer::noteCompositeFinalLayouts(uint32_t imageIndex,
                                               uint32_t historyIndex) {
  if (imageIndex >= renderResources.frameCount())
    return;

  const FrameRenderTargets &targets = renderResources.frame(imageIndex);
  renderResources.noteLayout(targets.swapchainImage,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  renderResources.noteLayout(targets.colorBuffer,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  if (historyIndex < taaHistoryImages.size()) {
    renderResources.noteLayout(taaHistoryImages[historyIndex].get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
}

// G-buffer resource management

void VulkanRenderer::createGBuffer() {
  size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  VkFormat gb0Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb1Fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat gb2Fmt = VK_FORMAT_R8G8B8A8_UNORM;
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  const RenderQueueSharingInfo queueSharing =
      renderGraphicsComputeSharing(queues);

  if (renderResources.frameCount() != count)
    renderResources.resizeFrames(count);

  gBuffer0Images.resize(count);
  gBuffer0Views.resize(count);
  gBuffer1Images.resize(count);
  gBuffer1Views.resize(count);
  gBuffer2Images.resize(count);
  gBuffer2Views.resize(count);
  gBufferDepthImages.resize(count);
  gBufferDepthViews.resize(count);
  gBufferFramebuffers.resize(count, VK_NULL_HANDLE);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  auto makeColorImg = [&](AllocatedImage &img, ImageViewHandle &view,
                          VkFormat fmt) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = fmt;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    applyRenderQueueSharing(ci, queueSharing);
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer color view");
    view = ImageViewHandle(dev, v);
  };

  auto makeDepthImg = [&](AllocatedImage &img, ImageViewHandle &view) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = gBufferDepthFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    applyRenderQueueSharing(ci, queueSharing);
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer depth image");
    img = AllocatedImage(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = gBufferDepthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkImageView v;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer depth view");
    view = ImageViewHandle(dev, v);
  };

  for (size_t i = 0; i < count; i++) {
    makeColorImg(gBuffer0Images[i], gBuffer0Views[i], gb0Fmt);
    makeColorImg(gBuffer1Images[i], gBuffer1Views[i], gb1Fmt);
    makeColorImg(gBuffer2Images[i], gBuffer2Views[i], gb2Fmt);
    makeDepthImg(gBufferDepthImages[i], gBufferDepthViews[i]);

    renderResources.registerImage(
        {"gBuffer0[" + std::to_string(i) + "]", gBuffer0Images[i].get(),
         gb0Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBuffer1[" + std::to_string(i) + "]", gBuffer1Images[i].get(),
         gb1Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBuffer2[" + std::to_string(i) + "]", gBuffer2Images[i].get(),
         gb2Fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});
    renderResources.registerImage(
        {"gBufferDepth[" + std::to_string(i) + "]",
         gBufferDepthImages[i].get(), gBufferDepthFormat,
         renderImageAspectMask(gBufferDepthFormat), 1, 1,
         VK_IMAGE_LAYOUT_UNDEFINED, queueSharing.mode, graphicsFamily});

    FrameRenderTargets &targets = renderResources.frame(i);
    targets.gBuffer0 = gBuffer0Images[i].get();
    targets.gBuffer1 = gBuffer1Images[i].get();
    targets.gBuffer2 = gBuffer2Images[i].get();
    targets.gBufferDepth = gBufferDepthImages[i].get();

    std::array<VkImageView, 4> attachments = {
        gBuffer0Views[i].get(), gBuffer1Views[i].get(), gBuffer2Views[i].get(),
        gBufferDepthViews[i].get()};
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getGBufferRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments = attachments.data();
    fbci.width = swapchain.getExtent().width;
    fbci.height = swapchain.getExtent().height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &gBufferFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create G-buffer framebuffer");
  }

  std::vector<VkImageView> gb0v, gb1v, gb2v, depv, litv;
  gb0v.reserve(count);
  gb1v.reserve(count);
  gb2v.reserve(count);
  depv.reserve(count);
  litv.reserve(count);
  for (size_t i = 0; i < count; i++) {
    gb0v.push_back(gBuffer0Views[i].get());
    gb1v.push_back(gBuffer1Views[i].get());
    gb2v.push_back(gBuffer2Views[i].get());
    depv.push_back(gBufferDepthViews[i].get());
    litv.push_back(litViews[i].get());
  }
  // SSGI views are a temporal history ring. DescriptorManager produces
  // historyCount * swapCount G-buffer sets indexed
  // (historyIndex * swapCount + i); per history index H, binding 5 =
  // SsgiPass history view H.
  std::vector<VkImageView> ssgv = ssgiPass.views();
  const std::vector<VkImageView> ssaov = ssaoPass.views();
  descriptorManager.recreateGBufferSets(device.getLogicalDevice(), gb0v, gb1v,
                                        gb2v, depv, litv, ssgv,
                                        ssaov,
                                        textureManager.getTextureSampler(),
                                        ssaoPass.sampler());
  {
    std::vector<VkImageView> ssgiHv = ssgiPass.views();
    descriptorManager.recreateSsgiPrevSets(device.getLogicalDevice(), ssgiHv,
                                           ssgiPass.sampler());
  }
}

void VulkanRenderer::cleanupGBuffer() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : gBufferFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  gBufferFramebuffers.clear();
  gBufferDepthViews.clear();
  gBufferDepthImages.clear();
  gBuffer2Views.clear();
  gBuffer2Images.clear();
  gBuffer1Views.clear();
  gBuffer1Images.clear();
  gBuffer0Views.clear();
  gBuffer0Images.clear();
}

// Lit-buffer resource management (post-PBR HDR, sampled by SSR composite)

void VulkanRenderer::createLitResources() {
  VkDevice dev = device.getLogicalDevice();
  size_t count = swapchain.getImageCount();
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);

  if (renderResources.frameCount() != count)
    renderResources.resizeFrames(count);

  litImages.clear();
  litViews.clear();
  litFramebuffers.clear();
  litImages.reserve(count);
  litViews.reserve(count);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  for (size_t i = 0; i < count; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = litFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit image");
    litImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = litImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = litFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit view");
    litViews.emplace_back(dev, v);

    renderResources.registerImage(
        {"lit[" + std::to_string(i) + "]", litImages.back().get(), litFormat,
         VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_SHARING_MODE_EXCLUSIVE, graphicsFamily});
    renderResources.frame(i).lit = litImages.back().get();

  }

  // One sampler shared across all swapchain images (linear, clamp).
  if (litSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &litSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create lit sampler");
  }
}

void VulkanRenderer::createLitFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  const size_t count = swapchain.getImageCount();
  if (litViews.size() != count || gBufferDepthViews.size() != count)
    throw std::runtime_error("Lit framebuffer attachments are not ready");

  cleanupLitFramebuffers();
  litFramebuffers.assign(count, VK_NULL_HANDLE);
  for (size_t i = 0; i < count; ++i) {
    std::array<VkImageView, 2> attachments = {
        litViews[i].get(), gBufferDepthViews[i].get()};
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPassManager.getLitRenderPass();
    fbci.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbci.pAttachments = attachments.data();
    fbci.width = swapchain.getExtent().width;
    fbci.height = swapchain.getExtent().height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &litFramebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create lit framebuffer");
  }
}

void VulkanRenderer::cleanupLitFramebuffers() {
  VkDevice dev = device.getLogicalDevice();
  for (VkFramebuffer fb : litFramebuffers)
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, fb, nullptr);
  litFramebuffers.clear();
}

void VulkanRenderer::cleanupLitResources() {
  VkDevice dev = device.getLogicalDevice();
  cleanupLitFramebuffers();
  litViews.clear();
  litImages.clear();
  if (litSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, litSampler, nullptr);
    litSampler = VK_NULL_HANDLE;
  }
}

// TAA history images. The ring is one larger than the max frames-in-flight,
// so a frame can sample the previous logical frame without another in-flight
// frame overwriting that image too early. Format matches litFormat so reads
// from history feed straight into the same HDR-precision pipeline. Usage:
// SAMPLED (read as history-prev) + COLOR_ATTACHMENT (written as history-curr).
void VulkanRenderer::createTaaResources() {
  VkDevice dev = device.getLogicalDevice();
  const QueueFamilyIndices queues = device.getQueueFamilies();
  const uint32_t graphicsFamily = static_cast<uint32_t>(queues.graphicsFamily);
  taaHistoryImages.clear();
  taaHistoryViews.clear();
  const size_t historyCount = MAX_FRAMES_DRAWS + 1;
  taaHistoryImages.reserve(historyCount);
  taaHistoryViews.reserve(historyCount);

  VmaAllocationCreateInfo aci = {};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  for (size_t i = 0; i < historyCount; ++i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchain.getExtent().width, swapchain.getExtent().height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = litFormat;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage rawImg = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    if (vmaCreateImage(device.getAllocator(), &ci, &aci, &rawImg, &alloc,
                       nullptr) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA history image");
    taaHistoryImages.emplace_back(device.getAllocator(), rawImg, alloc);

    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = taaHistoryImages.back().get();
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = litFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &vci, nullptr, &v) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA history view");
    taaHistoryViews.emplace_back(dev, v);

    renderResources.registerImage(
        {"taaHistory[" + std::to_string(i) + "]",
         taaHistoryImages.back().get(), litFormat, VK_IMAGE_ASPECT_COLOR_BIT,
         1, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_SHARING_MODE_EXCLUSIVE,
         graphicsFamily});
  }

  // Transition both to SHADER_READ_ONLY_OPTIMAL so the first frame's
  // sampler bind doesn't read undefined data. The composite render pass
  // (when wired up next session) will declare initialLayout=UNDEFINED on
  // the history attachment with loadOp=DONT_CARE, so the transition back
  // to COLOR_ATTACHMENT_OPTIMAL is handled by the render pass itself.
  VkCommandBuffer cmd = beginCommandBuffer(dev, device.getGraphicsCommandPool());
  for (size_t i = 0; i < historyCount; ++i) {
    renderResources.transition(
        cmd, taaHistoryImages[i].get(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_ACCESS_SHADER_READ_BIT);
  }
  endAndSubmitCommandBuffer(dev, device.getGraphicsCommandPool(),
                            device.getGraphicsQueue(), cmd);

  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that slide just past the screen edge
  // sample the edge texel rather than wrapping; the shader's explicit
  // [0,1] bounds check handles true disocclusion.
  if (taaSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &taaSampler) != VK_SUCCESS)
      throw std::runtime_error("Failed to create TAA sampler");
  }

  // Fresh allocation → history is undefined → don't blend the first frame.
  // Shared with SSGI: ssgi.frag reads scene.taaParams.w (==taaHistoryValid),
  // so dropping this also drops the SSGI temporal blend on the post-resize
  // frame. SsgiPass is recreated alongside createTaaResources on every
  // swapchain recreate; both temporal history rings invalidate together.
  taaHistoryValid = false;
}

void VulkanRenderer::cleanupTaaResources() {
  // ImageViewHandle / AllocatedImage destructors handle vkDestroyImageView /
  // vmaDestroyImage. We just clear the vectors.
  taaHistoryViews.clear();
  taaHistoryImages.clear();
  if (taaSampler != VK_NULL_HANDLE) {
    vkDestroySampler(device.getLogicalDevice(), taaSampler, nullptr);
    taaSampler = VK_NULL_HANDLE;
  }
}

// IBL resource management

void VulkanRenderer::initIBL() {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &props);

  // Sampler for irradiance / prefiltered / brdf-lut / skybox (mip-capable)
  VkSamplerCreateInfo iblCi = {};
  iblCi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  iblCi.magFilter = VK_FILTER_LINEAR;
  iblCi.minFilter = VK_FILTER_LINEAR;
  iblCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  iblCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  iblCi.minLod = 0.0f;
  iblCi.maxLod = static_cast<float>(IBL_PREFILTER_MIPS);
  iblCi.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &iblCi, nullptr,
                      &iblSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create IBL sampler");

  // Sampler for 4x4 SSAO noise (tiled nearest)
  VkSamplerCreateInfo noiseCi = {};
  noiseCi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  noiseCi.magFilter = VK_FILTER_NEAREST;
  noiseCi.minFilter = VK_FILTER_NEAREST;
  noiseCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  noiseCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  noiseCi.maxAnisotropy = 1.0f;
  if (vkCreateSampler(device.getLogicalDevice(), &noiseCi, nullptr,
                      &ssaoNoiseSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create SSAO noise sampler");

  spdlog::info(
      "IBL: loading skybox and generating irradiance / prefiltered maps...");
  activeEnvironmentHdr = "Resources/HDRIs/kloppenheim_02_1k.hdr";
  iblSkyboxImageIndex =
      textureManager.loadSkybox(activeEnvironmentHdr, device, descriptorManager);
  irradianceImageIndex =
      textureManager.createIrradianceMap(iblSkyboxImageIndex, device);
  prefilteredEnvImageIndex =
      textureManager.createPrefilteredEnvMap(iblSkyboxImageIndex, device);
  brdfLutImageIndex = textureManager.loadBrdfLut(device);
  ssaoNoiseImageIndex = textureManager.createSsaoNoiseTexture(device);
  spdlog::info("IBL: done.");

  descriptorManager.updateIblDescriptors(
      device.getLogicalDevice(),
      textureManager.getImageView(irradianceImageIndex), iblSampler,
      textureManager.getImageView(prefilteredEnvImageIndex),
      textureManager.getImageView(brdfLutImageIndex),
      textureManager.getImageView(ssaoNoiseImageIndex), ssaoNoiseSampler,
      textureManager.getImageView(iblSkyboxImageIndex));
}

void VulkanRenderer::cleanupIBL() {
  VkDevice dev = device.getLogicalDevice();
  if (ssaoNoiseSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, ssaoNoiseSampler, nullptr);
    ssaoNoiseSampler = VK_NULL_HANDLE;
  }
  if (iblSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, iblSampler, nullptr);
    iblSampler = VK_NULL_HANDLE;
  }
}

// Cleanup

void VulkanRenderer::addInstancedModel(
    int modelId, const std::vector<glm::mat4> &transforms) {
  InstancedDrawable drawable;
  drawable.modelId = modelId;
  drawable.instances.reserve(transforms.size());
  for (const glm::mat4 &t : transforms) {
    InstanceData id{};
    id.model = t;
    id.normal = glm::transpose(glm::inverse(t));
    drawable.instances.push_back(id);
  }

  // Compute a world-space bounding sphere covering every instance.
  // Used in recordCommands to cull/LOD the entire batch.
  MeshModel *mdl = modelManager.getModel(modelId);
  float modelRadius = mdl ? mdl->boundingRadius : 1.0f;
  glm::vec3 modelCenter = mdl ? mdl->boundingCenter : glm::vec3(0.0f);
  glm::vec3 minP(FLT_MAX), maxP(-FLT_MAX);
  for (const InstanceData &id : drawable.instances) {
    glm::vec3 wc = glm::vec3(id.model * glm::vec4(modelCenter, 1.0f));
    minP = glm::min(minP, wc);
    maxP = glm::max(maxP, wc);
  }
  drawable.groupCenter = (minP + maxP) * 0.5f;
  // Largest distance from group center to any instance, plus the model's own
  // radius scaled by the largest instance scale.
  float maxInstScale = 0.0f;
  float maxInstDist = 0.0f;
  for (const InstanceData &id : drawable.instances) {
    glm::vec3 wc = glm::vec3(id.model * glm::vec4(modelCenter, 1.0f));
    maxInstDist = glm::max(maxInstDist, glm::length(wc - drawable.groupCenter));
    float s = glm::max(glm::length(glm::vec3(id.model[0])),
                       glm::max(glm::length(glm::vec3(id.model[1])),
                                glm::length(glm::vec3(id.model[2]))));
    maxInstScale = glm::max(maxInstScale, s);
  }
  drawable.groupRadius = maxInstDist + modelRadius * maxInstScale;

  VkDeviceSize bufferSize = sizeof(InstanceData) * drawable.instances.size();
  createBuffer(device.getAllocator(), bufferSize,
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
               &drawable.instanceBuffer);

  uploadToAllocation(device.getAllocator(),
                     drawable.instanceBuffer.getAllocation(),
                     drawable.instances.data(), bufferSize);

  instancedDrawables.push_back(std::move(drawable));
}

void VulkanRenderer::cleanup() {
  vkDeviceWaitIdle(device.getLogicalDevice());

  metrics.printReport(device.getAllocator());

  renderResources.clear();
  instancedDrawables.clear(); // AllocatedBuffer RAII destroys GPU buffers
  imguiLayer.cleanup(device.getLogicalDevice());
  modelManager.cleanup();
  textureManager.cleanup(device.getLogicalDevice(), device.getAllocator());
  autoExposurePass.cleanup();
  compositePass.cleanup();
  gpuDrivenGBufferPass.cleanup();
  materialProbePass.cleanup();
  cleanupLitFramebuffers();
  cleanupGBuffer();
  ssaoPass.cleanup();
  bloomPass.cleanup();
  ssgiPass.cleanup();
  cleanupLitResources();
  cleanupTaaResources();
  cleanupIBL();
  shadowPass.cleanup();

  descriptorManager.cleanup(device.getLogicalDevice(), device.getAllocator(),
                            swapchain.getImageCount());
  swapchain.cleanup(device.getLogicalDevice(), device.getAllocator());

  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    vkDestroySemaphore(device.getLogicalDevice(), imageAvailable[i], nullptr);
    vkDestroyFence(device.getLogicalDevice(), drawFences[i], nullptr);
  }
  for (VkSemaphore s : renderFinished)
    vkDestroySemaphore(device.getLogicalDevice(), s, nullptr);
  renderFinished.clear();
  for (VkSemaphore s : asyncComputeTimeline)
    vkDestroySemaphore(device.getLogicalDevice(), s, nullptr);
  asyncComputeTimeline.clear();
  asyncComputeTimelineValue.clear();
  cleanupAsyncFrameCommandBuffers();
  cleanupThreadedCommandResources();

  pipeline.cleanup(device.getLogicalDevice());
  pipelineCache.cleanup(device.getLogicalDevice());
  renderPassManager.cleanup(device.getLogicalDevice());
  metrics.cleanup(device.getLogicalDevice());
  device.cleanup();
}

VulkanRenderer::~VulkanRenderer() {}

// Command recording

static void extractFrustumPlanes(const glm::mat4 &vp, glm::vec4 planes[6]) {
  // Gribb-Hartmann: GLM is column-major, so transpose to access rows as columns
  glm::mat4 t = glm::transpose(vp);
  planes[0] = t[3] + t[0]; // left
  planes[1] = t[3] - t[0]; // right
  planes[2] = t[3] + t[1]; // bottom
  planes[3] = t[3] - t[1]; // top
  planes[4] = t[3] + t[2]; // near
  planes[5] = t[3] - t[2]; // far
  for (int i = 0; i < 6; i++)
    planes[i] /= glm::length(glm::vec3(planes[i]));
}

static bool sphereInFrustum(const glm::vec4 planes[6], glm::vec3 center,
                            float radius) {
  for (int i = 0; i < 6; i++)
    if (glm::dot(glm::vec3(planes[i]), center) + planes[i].w < -radius)
      return false;
  return true;
}

static glm::uvec4 materialTextureIndices0(const Material &mat) {
  return glm::uvec4(static_cast<uint32_t>(mat.albedoTextureId),
                    static_cast<uint32_t>(mat.normalTextureId),
                    static_cast<uint32_t>(mat.metallicTextureId),
                    static_cast<uint32_t>(mat.roughnessTextureId));
}

static glm::uvec4 materialTextureIndices1(const Material &mat) {
  const uint32_t materialFlags =
      (mat.isCloth ? 1u : 0u) | (mat.alphaMasked ? 2u : 0u) |
      (mat.alphaBlended ? 4u : 0u);
  const uint32_t alphaCutoff255 = static_cast<uint32_t>(
      glm::clamp(mat.alphaCutoff, 0.0f, 1.0f) * 255.0f + 0.5f);
  return glm::uvec4(static_cast<uint32_t>(mat.aoTextureId), materialFlags,
                    alphaCutoff255, 0u);
}

static void sortGBufferDrawItemsForState(std::vector<GBufferDrawItem> &items) {
  std::stable_sort(items.begin(), items.end(),
                   [](const GBufferDrawItem &a, const GBufferDrawItem &b) {
                     if (a.kind != b.kind)
                       return static_cast<int>(a.kind) <
                              static_cast<int>(b.kind);
                     if (a.cullMode != b.cullMode)
                       return a.cullMode < b.cullMode;
                     if (a.mesh != b.mesh)
                       return std::less<const Mesh *>{}(a.mesh, b.mesh);
                     return a.lod < b.lod;
                   });

  for (size_t i = 0; i < items.size(); ++i) {
    items[i].materialId = static_cast<uint32_t>(i);
    items[i].transformId = static_cast<uint32_t>(i);
  }
}

struct GBufferStateStats {
  uint32_t pipelineBinds = 0;
  uint32_t dynamicStateChanges = 0;
};

struct TransparentDrawItem {
  GBufferDrawKind kind = GBufferDrawKind::Mesh;
  const Mesh *mesh = nullptr;
  int lod = 0;
  ModelPushConstants push{};
  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
  VkBuffer instanceBuffer = VK_NULL_HANDLE;
  const std::vector<InstanceData> *instances = nullptr;
  uint32_t instanceCount = 1;
  uint32_t indexCount = 0;
  float distanceSq = 0.0f;
};

static GBufferStateStats estimateGBufferStateStats(
    const std::vector<GBufferDrawItem> &items, uint32_t partitionCount) {
  GBufferStateStats stats{};
  partitionCount = std::max(1u, partitionCount);

  for (uint32_t partition = 0; partition < partitionCount; ++partition) {
    const size_t begin =
        (items.size() * size_t(partition)) / size_t(partitionCount);
    const size_t end =
        (items.size() * size_t(partition + 1)) / size_t(partitionCount);

    GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
    bool pipelineBound = false;
    VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
    bool cullValid = false;

    for (size_t i = begin; i < end; ++i) {
      const GBufferDrawItem &item = items[i];
      if (!pipelineBound || boundKind != item.kind) {
        stats.pipelineBinds++;
        boundKind = item.kind;
        pipelineBound = true;
        cullValid = false;
      }

      if (!cullValid || item.cullMode != lastCull) {
        stats.dynamicStateChanges++;
        lastCull = item.cullMode;
        cullValid = true;
      }
    }
  }

  return stats;
}

void VulkanRenderer::recordCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin recording command buffer");

  metrics.resetGpuQueries(cmd);

  vkdbgBeginLabel(cmd, "CSM Shadow Pass", 1.0f, 0.4f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  shadowPass.recordCsm(cmd, rootNode, modelManager, pipeline, descriptorManager,
                       sceneUbo, imguiShadowFrontFaceCull,
                       imguiCullShadowCasters);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  vkdbgEndLabel(cmd);

  vkdbgBeginLabel(cmd, "Point Shadow Pass", 1.0f, 0.6f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  shadowPass.recordPoint(cmd, rootNode, modelManager, pipeline,
                         descriptorManager, imguiShadowFrontFaceCull);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  vkdbgEndLabel(cmd);

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};

  // Temporal history indices. Rings are larger than the number of frames in
  // flight, so an in-flight frame's previous-history read is not overwritten
  // by a later CPU submission.
  uint32_t ssgiHistoryIndex = ssgiPass.historyIndex(taaFrameCounter);
  uint32_t taaHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % taaHistoryViews.size());

  recordGBufferPass(cmd, currentImage, viewport, scissor);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
  ssgiPass.record(cmd, ssgiHistoryIndex,
                  descriptorManager.getVPSet(currentImage),
                  descriptorManager.getGBufferSet(ssgiHistoryIndex,
                                                  currentImage),
                  descriptorManager.getSsgiPrevSet(ssgiHistoryIndex),
                  pipeline.getSsgiPipeline(), pipeline.getSsgiLayout());
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);

  // --- Lit pass (PBR + IBL + SSAO + FXAA + fog → litBuffer) ---
  {
    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getLitRenderPass();
    rpbi.framebuffer = litFramebuffers[currentImage];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkdbgBeginLabel(cmd, "Deferred Lighting (PBR+IBL+SSAO)", 0.1f, 0.8f,
                    0.3f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getLitPipeline());
    std::array<VkDescriptorSet, 2> litSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getLitLayout(), 0, 2, litSets.data(), 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    recordTransparentPass(cmd, currentImage, viewport, scissor);
    vkCmdEndRenderPass(cmd);
    noteLitFinalLayout(currentImage);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkdbgEndLabel(cmd);
  }

  // --- Bloom pyramid (compute) ---
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);
  const bool bloomRecorded =
      bloomPass.record(cmd, currentImage, litImages, imguiEnableBloom,
                       imguiDebugMode, imguiBloomThreshold, imguiBloomRadius,
                       imguiBloomIntensity);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);

  // --- Auto-exposure (compute) ---
  // Runs between lit and composite: lit is now in SHADER_READ_ONLY_OPTIMAL
  // (the render pass's finalLayout). Pass 1 histograms it; pass 2 reduces
  // and writes one float into a host-visible buffer that next frame's CPU
  // reads to drive eye adaptation.
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);
  autoExposurePass.record(cmd, currentImage, currentFrame, litImages,
                          autoExpEnabled, bloomRecorded);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);

  const size_t compositeFbIdx =
      compositePass.framebufferIndex(taaHistoryIndex, currentImage);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
  compositePass.record(
      cmd, currentImage, taaHistoryIndex,
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      descriptorManager.getInputSet(currentImage),
      descriptorManager.getTaaSet(compositeFbIdx),
      pipeline.getDeferredPipeline(), pipeline.getDeferredLayout(),
      pipeline.getSecondPipeline(), pipeline.getSecondLayout());
  noteCompositeFinalLayouts(currentImage, taaHistoryIndex);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  recordImGuiCommands(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to stop recording command buffer");
}

void VulkanRenderer::recordGBufferCommands(uint32_t currentImage) {
  VkCommandBuffer cmd = swapchain.getCommandBuffer(currentImage);
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin G-buffer command buffer");

  metrics.resetGpuQueries(cmd);

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};
  recordGBufferPass(cmd, currentImage, viewport, scissor);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end G-buffer command buffer");
}

void VulkanRenderer::recordShadowCommands(VkCommandBuffer cmd) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin shadow command buffer");

  vkdbgBeginLabel(cmd, "CSM Shadow Pass", 1.0f, 0.4f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  shadowPass.recordCsm(cmd, rootNode, modelManager, pipeline, descriptorManager,
                       sceneUbo, imguiShadowFrontFaceCull,
                       imguiCullShadowCasters);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Shadow);
  vkdbgEndLabel(cmd);

  vkdbgBeginLabel(cmd, "Point Shadow Pass", 1.0f, 0.6f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  shadowPass.recordPoint(cmd, rootNode, modelManager, pipeline,
                         descriptorManager, imguiShadowFrontFaceCull);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::PointShadow);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end shadow command buffer");
}

void VulkanRenderer::recordSsaoComputeCommands(VkCommandBuffer cmd,
                                               uint32_t currentImage) {
  const uint32_t ssgiHistoryIndex = ssgiPass.historyIndex(taaFrameCounter);
  ssaoPass.recordCommands(
      cmd, currentImage, descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage));
}

void VulkanRenderer::recordPostCommands(VkCommandBuffer cmd,
                                        uint32_t currentImage) {
  vkResetCommandBuffer(cmd, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin post command buffer");

  VkViewport viewport = {};
  viewport.width = static_cast<float>(swapchain.getExtent().width);
  viewport.height = static_cast<float>(swapchain.getExtent().height);
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {{0, 0}, swapchain.getExtent()};

  uint32_t ssgiHistoryIndex = ssgiPass.historyIndex(taaFrameCounter);
  uint32_t taaHistoryIndex =
      static_cast<uint32_t>(taaFrameCounter % taaHistoryViews.size());

  materialProbePass.record(
      cmd, static_cast<uint32_t>(currentFrame),
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      materialProbePixel);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);
  ssgiPass.record(cmd, ssgiHistoryIndex,
                  descriptorManager.getVPSet(currentImage),
                  descriptorManager.getGBufferSet(ssgiHistoryIndex,
                                                  currentImage),
                  descriptorManager.getSsgiPrevSet(ssgiHistoryIndex),
                  pipeline.getSsgiPipeline(), pipeline.getSsgiLayout());
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::SSGI);

  // --- Lit pass (PBR + IBL + async SSAO + FXAA + fog -> litBuffer) ---
  {
    VkClearValue clear = {};
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPassManager.getLitRenderPass();
    rpbi.framebuffer = litFramebuffers[currentImage];
    rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkdbgBeginLabel(cmd, "Deferred Lighting (PBR+IBL+SSAO)", 0.1f, 0.8f,
                    0.3f);
    metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getLitPipeline());
    std::array<VkDescriptorSet, 2> litSets = {
        descriptorManager.getVPSet(currentImage),
        descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getLitLayout(), 0, 2, litSets.data(), 0,
                            nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
    recordTransparentPass(cmd, currentImage, viewport, scissor);
    vkCmdEndRenderPass(cmd);
    noteLitFinalLayout(currentImage);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Lit);
    vkdbgEndLabel(cmd);
  }

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);
  const bool bloomRecorded =
      bloomPass.record(cmd, currentImage, litImages, imguiEnableBloom,
                       imguiDebugMode, imguiBloomThreshold, imguiBloomRadius,
                       imguiBloomIntensity);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Bloom);

  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);
  autoExposurePass.record(cmd, currentImage, currentFrame, litImages,
                          autoExpEnabled, bloomRecorded);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::AutoExposure);

  const size_t compositeFbIdx =
      compositePass.framebufferIndex(taaHistoryIndex, currentImage);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);
  compositePass.record(
      cmd, currentImage, taaHistoryIndex,
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getGBufferSet(ssgiHistoryIndex, currentImage),
      descriptorManager.getInputSet(currentImage),
      descriptorManager.getTaaSet(compositeFbIdx),
      pipeline.getDeferredPipeline(), pipeline.getDeferredLayout(),
      pipeline.getSecondPipeline(), pipeline.getSecondLayout());
  noteCompositeFinalLayouts(currentImage, taaHistoryIndex);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::Composite);

  vkdbgBeginLabel(cmd, "ImGui Pass", 0.9f, 0.8f, 0.1f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  recordImGuiCommands(cmd, currentImage);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::ImGui);
  vkdbgEndLabel(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end post command buffer");
}

void VulkanRenderer::recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                                       const VkViewport &viewport,
                                       const VkRect2D &scissor) {
  std::vector<GBufferDrawItem> drawItems;
  drawItems.reserve(128);

  glm::vec4 frustumPlanes[6];
  extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);

  auto appendNode = [&](auto &self, SceneNode *node) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        glm::mat4 model = node->getGlobalTransform();
        glm::vec3 wCenter =
            glm::vec3(model * glm::vec4(mdl->boundingCenter, 1.0f));
        float maxScale =
            glm::max(glm::length(glm::vec3(model[0])),
                     glm::max(glm::length(glm::vec3(model[1])),
                              glm::length(glm::vec3(model[2]))));
        if (sphereInFrustum(frustumPlanes, wCenter,
                            mdl->boundingRadius * maxScale)) {
          float camDist =
              glm::length(wCenter - glm::vec3(sceneUbo.cameraPosition));
          int lod = (camDist < imguiLodNear)  ? 0
                    : (camDist < imguiLodFar) ? 1
                                              : 2;

          ModelPushConstants basePush{};
          basePush.model = model;
          basePush.normal = node->getNormalMatrix();

          for (size_t k = 0; k < mdl->getMeshCount(); k++) {
            const Mesh *mesh = mdl->getMesh(k);
            glm::vec3 meshWCenter =
                glm::vec3(model * glm::vec4(mesh->boundingCenter, 1.0f));
            if (!sphereInFrustum(frustumPlanes, meshWCenter,
                                 mesh->boundingRadius * maxScale))
              continue;

            const Material &mat = mesh->getMaterial();
            if (mat.alphaBlended)
              continue;
            GBufferDrawItem item{};
            item.kind = GBufferDrawKind::Mesh;
            item.mesh = mesh;
            item.lod = std::min(lod, mesh->getLodCount() - 1);
            item.push = basePush;
            item.push.texIdx0 = materialTextureIndices0(mat);
            item.push.texIdx1 = materialTextureIndices1(mat);
            glm::vec3 radiusVec(mesh->boundingRadius * maxScale);
            item.aabbMin = meshWCenter - radiusVec;
            item.aabbMax = meshWCenter + radiusVec;
            item.cullMode =
                mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
            item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
            item.materialId = static_cast<uint32_t>(drawItems.size());
            item.transformId = static_cast<uint32_t>(drawItems.size());
            drawItems.push_back(item);
          }
        }
      }
    }
    for (auto &child : node->getChildren())
      self(self, child.get());
  };
  appendNode(appendNode, &rootNode);

  // Instanced drawables still become ordinary draw items here. The secondary
  // command buffer recorder switches to the instanced pipeline for these items.
  for (const InstancedDrawable &drawable : instancedDrawables) {
    MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                         drawable.groupRadius))
      continue;

    float groupDist =
        glm::length(drawable.groupCenter - glm::vec3(sceneUbo.cameraPosition));
    int instLod = (groupDist < imguiLodNear)  ? 0
                  : (groupDist < imguiLodFar) ? 1
                                              : 2;
    for (size_t k = 0; k < mdl->getMeshCount(); k++) {
      const Mesh *mesh = mdl->getMesh(k);
      const Material &mat = mesh->getMaterial();
      if (mat.alphaBlended)
        continue;
      GBufferDrawItem item{};
      item.kind = GBufferDrawKind::Instanced;
      item.mesh = mesh;
      item.lod = std::min(instLod, mesh->getLodCount() - 1);
      item.push.texIdx0 = materialTextureIndices0(mat);
      item.push.texIdx1 = materialTextureIndices1(mat);
      glm::vec3 radiusVec(drawable.groupRadius);
      item.aabbMin = drawable.groupCenter - radiusVec;
      item.aabbMax = drawable.groupCenter + radiusVec;
      item.cullMode = VK_CULL_MODE_BACK_BIT;
      item.instanceBuffer = drawable.instanceBuffer.get();
      item.instances = &drawable.instances;
      item.instanceCount = static_cast<uint32_t>(drawable.instances.size());
      item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
      item.materialId = static_cast<uint32_t>(drawItems.size());
      item.transformId = static_cast<uint32_t>(drawItems.size());
      drawItems.push_back(item);
    }
  }

  sortGBufferDrawItemsForState(drawItems);

  const uint32_t minGpuDrivenCandidates =
      static_cast<uint32_t>(std::max(0, imguiGpuDrivenMinCandidates));
  gpuDrivenGBufferPass.beginFrame(static_cast<uint32_t>(drawItems.size()));
  gpuDrivenGBufferPass.prepareFrame(
      currentImage, drawItems, frustumPlanes,
      sceneUbo.projection * sceneUbo.view, imguiGpuDrivenEnabled,
      imguiHzbCullingEnabled, minGpuDrivenCandidates);

  for (const GBufferDrawItem &item : drawItems)
    metrics.recordDrawCall(item.indexCount * item.instanceCount);

  std::array<VkClearValue, 4> clears{};
  clears[3].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rpbi = {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = renderPassManager.getGBufferRenderPass();
  rpbi.framebuffer = gBufferFramebuffers[currentImage];
  rpbi.renderArea = {{0, 0}, swapchain.getExtent()};
  rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
  rpbi.pClearValues = clears.data();

  vkdbgBeginLabel(cmd, "G-Buffer Pass", 0.1f, 0.4f, 1.0f);
  metrics.beginPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);

  const VkDescriptorSet vpSet = descriptorManager.getVPSet(currentImage);
  const VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
  const std::array<VkDescriptorSet, 2> graphicsSets = {vpSet, bindlessSet};
  const std::array<VkDescriptorSet, 2> instancedSets = {vpSet, bindlessSet};

  auto recordDrawCommands = [&](VkCommandBuffer drawCmd, size_t begin,
                                size_t end) {
    vkCmdSetViewport(drawCmd, 0, 1, &viewport);
    vkCmdSetScissor(drawCmd, 0, 1, &scissor);
    GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
    bool pipelineBound = false;
    VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
    bool cullValid = false;

    for (size_t i = begin; i < end; ++i) {
      const GBufferDrawItem &item = drawItems[i];
      if (!pipelineBound || boundKind != item.kind) {
        if (item.kind == GBufferDrawKind::Mesh) {
          vkCmdBindPipeline(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getGraphicsPipeline());
          vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.getGraphicsLayout(), 0, 2,
                                  graphicsSets.data(), 0, nullptr);
        } else {
          vkCmdBindPipeline(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getInstancedPipeline());
          vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.getInstancedLayout(), 0, 2,
                                  instancedSets.data(), 0, nullptr);
        }
        boundKind = item.kind;
        pipelineBound = true;
        cullValid = false;
      }

      if (!cullValid || item.cullMode != lastCull) {
        vkCmdSetCullMode(drawCmd, item.cullMode);
        lastCull = item.cullMode;
        cullValid = true;
      }

      VkBuffer vb[] = {item.mesh->getVertexBuffer()};
      VkDeviceSize off[] = {0};
      vkCmdBindVertexBuffers(drawCmd, 0, 1, vb, off);
      if (item.kind == GBufferDrawKind::Instanced) {
        VkDeviceSize instanceOff = 0;
        vkCmdBindVertexBuffers(drawCmd, 1, 1, &item.instanceBuffer,
                               &instanceOff);
      }
      vkCmdBindIndexBuffer(drawCmd, item.mesh->getIndexBuffer(item.lod), 0,
                           VK_INDEX_TYPE_UINT32);
      VkPipelineLayout layout =
          item.kind == GBufferDrawKind::Mesh ? pipeline.getGraphicsLayout()
                                             : pipeline.getInstancedLayout();
      vkCmdPushConstants(drawCmd, layout,
                         VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(ModelPushConstants), &item.push);
      vkCmdDrawIndexed(drawCmd, item.indexCount, item.instanceCount, 0, 0, 0);
    }
  };

  const bool canRecordThreaded =
      imguiThreadedGBufferEnabled && commandThreadPool &&
      threadedCommandWorkerCount > 0 &&
      currentFrame < static_cast<int>(threadedCommandFrames.size()) &&
      !threadedCommandFrames[currentFrame].empty() && !drawItems.empty();

  const bool useGpuDriven = gpuDrivenGBufferPass.canRecordIndirect(
      currentImage, imguiGpuDrivenEnabled, minGpuDrivenCandidates);

  if (useGpuDriven) {
    metrics.recordPipelineBind(3);      // cull compute + two G-buffer pipelines
    metrics.recordDynamicStateChange(2); // viewport + scissor
    gpuDrivenGBufferPass.recordIndirectGBuffer(
        cmd, currentImage, rpbi, viewport, scissor, vpSet, bindlessSet);
    noteGBufferFinalLayouts(currentImage);
    gpuDrivenGBufferPass.recordHzbBuild(
        cmd, currentImage, gBufferDepthImages, gBufferDepthFormat,
        imguiGpuDrivenEnabled, imguiHzbCullingEnabled,
        minGpuDrivenCandidates);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkdbgEndLabel(cmd);
    return;
  }

  if (!canRecordThreaded) {
    const GBufferStateStats stateStats =
        estimateGBufferStateStats(drawItems, 1);
    metrics.recordPipelineBind(stateStats.pipelineBinds);
    metrics.recordDynamicStateChange(stateStats.dynamicStateChanges);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    recordDrawCommands(cmd, 0, drawItems.size());
    vkCmdEndRenderPass(cmd);
    noteGBufferFinalLayouts(currentImage);
    gpuDrivenGBufferPass.recordHzbBuild(
        cmd, currentImage, gBufferDepthImages, gBufferDepthFormat,
        imguiGpuDrivenEnabled, imguiHzbCullingEnabled,
        minGpuDrivenCandidates);
    metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
    vkdbgEndLabel(cmd);
    return;
  }

  auto &frameResources = threadedCommandFrames[currentFrame];
  const uint32_t workerCount = std::min<uint32_t>(
      threadedCommandWorkerCount, static_cast<uint32_t>(drawItems.size()));
  const GBufferStateStats stateStats =
      estimateGBufferStateStats(drawItems, workerCount);
  metrics.recordPipelineBind(stateStats.pipelineBinds);
  metrics.recordDynamicStateChange(stateStats.dynamicStateChanges);
  std::vector<VkCommandBuffer> secondaries(workerCount, VK_NULL_HANDLE);

  for (uint32_t i = 0; i < workerCount; ++i) {
    ThreadCommandFrameResources &res = frameResources[i];
    vkResetCommandPool(device.getLogicalDevice(), res.pool, 0);
    secondaries[i] = res.gBufferSecondary;
  }

  const VkRenderPass gbufferPass = renderPassManager.getGBufferRenderPass();
  const VkFramebuffer gbufferFramebuffer = gBufferFramebuffers[currentImage];

  auto recordRange = [&](uint32_t workerIndex) {
    const size_t begin =
        (drawItems.size() * size_t(workerIndex)) / size_t(workerCount);
    const size_t end =
        (drawItems.size() * size_t(workerIndex + 1)) / size_t(workerCount);
    VkCommandBuffer secondary = secondaries[workerIndex];

    VkCommandBufferInheritanceInfo inheritance{};
    inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inheritance.renderPass = gbufferPass;
    inheritance.subpass = 0;
    inheritance.framebuffer = gbufferFramebuffer;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                      VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritance;
    if (vkBeginCommandBuffer(secondary, &beginInfo) != VK_SUCCESS)
      throw std::runtime_error("Failed to begin G-buffer secondary command buffer");

    recordDrawCommands(secondary, begin, end);
    if (vkEndCommandBuffer(secondary) != VK_SUCCESS)
      throw std::runtime_error("Failed to end G-buffer secondary command buffer");
  };

  commandThreadPool->dispatch(workerCount, recordRange);

  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  vkCmdExecuteCommands(cmd, workerCount, secondaries.data());
  vkCmdEndRenderPass(cmd);
  noteGBufferFinalLayouts(currentImage);
  gpuDrivenGBufferPass.recordHzbBuild(
      cmd, currentImage, gBufferDepthImages, gBufferDepthFormat,
      imguiGpuDrivenEnabled, imguiHzbCullingEnabled, minGpuDrivenCandidates);
  metrics.endPassTimestamp(cmd, PerformanceMetrics::GpuPass::GBuffer);
  vkdbgEndLabel(cmd);
}

void VulkanRenderer::recordTransparentPass(VkCommandBuffer cmd,
                                           uint32_t currentImage,
                                           const VkViewport &viewport,
                                           const VkRect2D &scissor) {
  std::vector<TransparentDrawItem> drawItems;
  drawItems.reserve(32);

  glm::vec4 frustumPlanes[6];
  extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, frustumPlanes);
  const glm::vec3 cameraPos = glm::vec3(sceneUbo.cameraPosition);

  auto appendNode = [&](auto &self, SceneNode *node) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *mdl = modelManager.getModel(node->getModelId());
      if (mdl) {
        glm::mat4 model = node->getGlobalTransform();
        glm::vec3 wCenter =
            glm::vec3(model * glm::vec4(mdl->boundingCenter, 1.0f));
        float maxScale =
            glm::max(glm::length(glm::vec3(model[0])),
                     glm::max(glm::length(glm::vec3(model[1])),
                              glm::length(glm::vec3(model[2]))));
        if (sphereInFrustum(frustumPlanes, wCenter,
                            mdl->boundingRadius * maxScale)) {
          float camDist = glm::length(wCenter - cameraPos);
          int lod = (camDist < imguiLodNear)  ? 0
                    : (camDist < imguiLodFar) ? 1
                                              : 2;

          ModelPushConstants basePush{};
          basePush.model = model;
          basePush.normal = node->getNormalMatrix();

          for (size_t k = 0; k < mdl->getMeshCount(); ++k) {
            const Mesh *mesh = mdl->getMesh(k);
            const Material &mat = mesh->getMaterial();
            if (!mat.alphaBlended)
              continue;

            glm::vec3 meshWCenter =
                glm::vec3(model * glm::vec4(mesh->boundingCenter, 1.0f));
            if (!sphereInFrustum(frustumPlanes, meshWCenter,
                                 mesh->boundingRadius * maxScale))
              continue;

            TransparentDrawItem item{};
            item.kind = GBufferDrawKind::Mesh;
            item.mesh = mesh;
            item.lod = std::min(lod, mesh->getLodCount() - 1);
            item.push = basePush;
            item.push.texIdx0 = materialTextureIndices0(mat);
            item.push.texIdx1 = materialTextureIndices1(mat);
            item.cullMode =
                mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
            item.indexCount =
                static_cast<uint32_t>(mesh->getIndexCount(item.lod));
            const glm::vec3 toCamera = meshWCenter - cameraPos;
            item.distanceSq = glm::dot(toCamera, toCamera);
            drawItems.push_back(item);
          }
        }
      }
    }
    for (auto &child : node->getChildren())
      self(self, child.get());
  };
  appendNode(appendNode, &rootNode);

  for (const InstancedDrawable &drawable : instancedDrawables) {
    MeshModel *mdl = modelManager.getModel(drawable.modelId);
    if (!mdl || drawable.instances.empty())
      continue;
    if (!sphereInFrustum(frustumPlanes, drawable.groupCenter,
                         drawable.groupRadius))
      continue;

    float groupDist = glm::length(drawable.groupCenter - cameraPos);
    int instLod = (groupDist < imguiLodNear)  ? 0
                  : (groupDist < imguiLodFar) ? 1
                                              : 2;
    for (size_t k = 0; k < mdl->getMeshCount(); ++k) {
      const Mesh *mesh = mdl->getMesh(k);
      const Material &mat = mesh->getMaterial();
      if (!mat.alphaBlended)
        continue;

      TransparentDrawItem item{};
      item.kind = GBufferDrawKind::Instanced;
      item.mesh = mesh;
      item.lod = std::min(instLod, mesh->getLodCount() - 1);
      item.push.texIdx0 = materialTextureIndices0(mat);
      item.push.texIdx1 = materialTextureIndices1(mat);
      item.cullMode =
          mat.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      item.instanceBuffer = drawable.instanceBuffer.get();
      item.instances = &drawable.instances;
      item.instanceCount = static_cast<uint32_t>(drawable.instances.size());
      item.indexCount = static_cast<uint32_t>(mesh->getIndexCount(item.lod));
      const glm::vec3 toCamera = drawable.groupCenter - cameraPos;
      item.distanceSq = glm::dot(toCamera, toCamera);
      drawItems.push_back(item);
    }
  }

  if (drawItems.empty())
    return;

  std::stable_sort(drawItems.begin(), drawItems.end(),
                   [](const TransparentDrawItem &a,
                      const TransparentDrawItem &b) {
                     return a.distanceSq > b.distanceSq;
                   });

  vkdbgBeginLabel(cmd, "Transparent Forward Pass", 0.1f, 0.65f, 0.9f);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  const std::array<VkDescriptorSet, 2> sets = {
      descriptorManager.getVPSet(currentImage),
      descriptorManager.getBindlessSet()};

  GBufferDrawKind boundKind = GBufferDrawKind::Mesh;
  bool pipelineBound = false;
  VkCullModeFlags lastCull = VK_CULL_MODE_BACK_BIT;
  bool cullValid = false;

  for (const TransparentDrawItem &item : drawItems) {
    if (!pipelineBound || boundKind != item.kind) {
      VkPipeline pipe = item.kind == GBufferDrawKind::Mesh
                            ? pipeline.getTransparentPipeline()
                            : pipeline.getTransparentInstancedPipeline();
      VkPipelineLayout layout = item.kind == GBufferDrawKind::Mesh
                                    ? pipeline.getTransparentLayout()
                                    : pipeline.getTransparentInstancedLayout();
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                              static_cast<uint32_t>(sets.size()), sets.data(),
                              0, nullptr);
      boundKind = item.kind;
      pipelineBound = true;
      cullValid = false;
    }

    if (!cullValid || item.cullMode != lastCull) {
      vkCmdSetCullMode(cmd, item.cullMode);
      lastCull = item.cullMode;
      cullValid = true;
    }

    VkBuffer vb[] = {item.mesh->getVertexBuffer()};
    VkDeviceSize off[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
    if (item.kind == GBufferDrawKind::Instanced) {
      VkDeviceSize instanceOff = 0;
      vkCmdBindVertexBuffers(cmd, 1, 1, &item.instanceBuffer, &instanceOff);
    }
    vkCmdBindIndexBuffer(cmd, item.mesh->getIndexBuffer(item.lod), 0,
                         VK_INDEX_TYPE_UINT32);

    VkPipelineLayout layout = item.kind == GBufferDrawKind::Mesh
                                  ? pipeline.getTransparentLayout()
                                  : pipeline.getTransparentInstancedLayout();
    vkCmdPushConstants(cmd, layout,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(ModelPushConstants), &item.push);
    vkCmdDrawIndexed(cmd, item.indexCount, item.instanceCount, 0, 0, 0);
    metrics.recordDrawCall(item.indexCount * item.instanceCount);
  }

  vkdbgEndLabel(cmd);
}

// ImGui integration

void VulkanRenderer::setImGuiCameraInfo(glm::vec3 pos, float speed) {
  imguiCameraPos = pos;
  imguiCameraSpeed = speed;
}

void VulkanRenderer::setCameraPresetCallback(CameraPresetCallback callback) {
  cameraPresetCallback = std::move(callback);
}

void VulkanRenderer::applySponzaReferencePreset() {
  // The saved raw-unit pose (-274, 46, -128) predates baked node transforms;
  // the world is metric now (Sponza root scale 0.008), so the same framing
  // is the old position times 0.008.
  const glm::vec3 cameraPosition(-2.19f, 0.37f, -1.02f);
  constexpr float cameraYaw = 25.0f;
  constexpr float cameraPitch = 3.0f;
  constexpr float cameraSpeed = 5.0f;

  imguiCameraPos = cameraPosition;
  imguiCameraSpeed = cameraSpeed;
  if (cameraPresetCallback)
    cameraPresetCallback(cameraPosition, cameraYaw, cameraPitch, cameraSpeed);

  imguiCameraFov = 65.0f;
  imguiDrawDistance = 200.0f;
  imguiLodNear = 15.0f;
  imguiLodFar = 45.0f;
  imguiPointShadowFar = 40.0f;
  imguiCsmFar = 80.0f;

  imguiDebugMode = 0;
  imguiUseGeomNormalOnly = false;
  imguiShadowFrontFaceCull = false;
  imguiCullShadowCasters = false;

  imguiSpecAAVariance = 1.60f;
  imguiSpecAAThreshold = 1.0f;
  imguiIblRoughnessFloor = 0.65f;
  imguiMinSurfaceRoughness = 0.60f;
  imguiSkyOcclusionFloor = 0.55f;
  imguiIblIntensity = 1.0f;
  imguiExposureEV = 0.0f;
  autoExpEnabled = true;
  autoExposurePass.resetAdaptation();

  imguiSsgiIntensity = 0.6f;
  imguiSsgiSamples = 8;
  imguiEnableSunDirect = true;
  imguiEnablePointLights = true;
  imguiEnableSpotLights = true;
  imguiEnableIblAmbient = true;
  imguiEnableSsgiBounce = true;
  imguiEnableBloom = true;
  imguiBloomThreshold = 1.2f;
  imguiBloomIntensity = 0.08f;
  imguiBloomRadius = 1.1f;
  imguiSsrEnabled = false;
  imguiTaaEnabled = true;
  imguiResponsiveTaa = true;
  imguiSharpness = 0.08f;
  imguiEdgeAA = 0.40f;
  imguiAlphaDither = 0.55f;
  imguiShadowSoftness = 1.35f;
  imguiContactGrounding = 0.55f;

  imguiDayNightEnable = false;
  imguiDayNightHour = 12.0f;
  imguiNormalStrength = 0.55f;
  imguiUseGeomNormalOnly = false;
  sceneUbo.directionalLight.direction =
      glm::vec4(glm::normalize(glm::vec3(-0.3f, -1.0f, 0.2f)), 0.0f);
  sceneUbo.directionalLight.colorIntensity =
      glm::vec4(1.0f, 0.93f, 0.78f, 2.35f);
  sceneUbo.debugMode = imguiDebugMode;
  sceneUbo.shadowParams.x = imguiPointShadowFar;
  sceneUbo.fogParams.x = imguiFogDensity;
  sceneUbo.fogParams.z = imguiFogClamp;

  rebuildProjection();
  shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                      imguiDrawDistance);
  shadowPass.updatePointShadowMatrices(sceneUbo);
  taaFrameCounter = 0;
  taaHistoryValid = false;
  taaHasLastCamera = false;
  cameraMovedThisFrame = true;
}

void VulkanRenderer::rebuildProjection() {
  float aspect = static_cast<float>(swapchain.getExtent().width) /
                 static_cast<float>(swapchain.getExtent().height);
  // Near=1.0 (not 0.1) because the scene is Crytek-scale (~3km long); a 0.1m
  // near plane burns most of the D32F depth range on the first metre and
  // leaves the rest of the hall z-fighting. 1m is below any meaningful
  // first-person clip distance at this scale.
  sceneUbo.projection = glm::perspective(glm::radians(imguiCameraFov), aspect,
                                         1.0f, imguiDrawDistance);
  sceneUbo.projection[1][1] *= -1;
  sceneUbo.invProj = glm::inverse(sceneUbo.projection);
  // Capture the un-jittered baseline — draw() will copy this into
  // sceneUbo.projection at the start of every frame before applying the
  // per-frame Halton offset, so the jitter does not accumulate across
  // frames as a random-walk drift.
  taaBaseProjection = sceneUbo.projection;
  // Any FOV/aspect/draw-distance change invalidates TAA history because the
  // reprojection math depends on the previous frame's projection.
  taaHistoryValid = false;
}

void VulkanRenderer::setFov(float fovDegrees) {
  imguiCameraFov = fovDegrees;
  rebuildProjection();
}

void VulkanRenderer::setDrawDistance(float dist) {
  imguiDrawDistance = dist;
  rebuildProjection();
}

void VulkanRenderer::recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex) {
  DebugUiContext ui{
      metrics,
      device.getAllocator(),
      sceneUbo,
      imguiCameraPos,
      imguiCameraSpeed,
      imguiCameraFov,
      imguiDrawDistance,
      imguiLodNear,
      imguiLodFar,
      imguiPointShadowFar,
      imguiFogDensity,
      imguiFogClamp,
      imguiDebugMode,
      imguiSpecAAVariance,
      imguiSpecAAThreshold,
      imguiIblRoughnessFloor,
      imguiShadowFrontFaceCull,
      imguiCullShadowCasters,
      imguiCsmFar,
      imguiIblIntensity,
      imguiExposureEV,
      imguiMinSurfaceRoughness,
      imguiSkyOcclusionFloor,
      imguiSsgiIntensity,
      imguiSsgiSamples,
      imguiEnableSunDirect,
      imguiEnablePointLights,
      imguiEnableSpotLights,
      imguiEnableIblAmbient,
      imguiEnableSsgiBounce,
      imguiEnableBloom,
      imguiBloomThreshold,
      imguiBloomIntensity,
      imguiBloomRadius,
      imguiSsrEnabled,
      imguiTaaEnabled,
      imguiResponsiveTaa,
      imguiPreferMailboxPresent,
      imguiSharpness,
      imguiEdgeAA,
      imguiAlphaDither,
      imguiShadowSoftness,
      imguiContactGrounding,
      imguiDayNightEnable,
      imguiDayNightSpeed,
      imguiDayNightHour,
      imguiNormalStrength,
      imguiUseGeomNormalOnly,
      imguiGpuDrivenEnabled,
      imguiHzbCullingEnabled,
      imguiGpuDrivenMinCandidates,
      imguiThreadedGBufferEnabled,
      autoExpEnabled,
      &availableSceneNames,
      activeSceneIndex,
      autoExposurePass.adaptedValue(),
      materialProbeResult,
      gpuDrivenGBufferPass.candidateCount(),
      gpuDrivenGBufferPass.meshCount(),
      gpuDrivenGBufferPass.lastFrameUsed(),
      threadedCommandWorkerCount,
      swapchain.getActivePresentMode(),
      [this] { rebuildProjection(); },
      [this] {
        shadowPass.updateLightSpaceMatrices(sceneUbo, imguiCsmFar,
                                            imguiDrawDistance);
      },
      [this] { shadowPass.updatePointShadowMatrices(sceneUbo); },
      [this] { taaHistoryValid = false; },
      [this] { applySponzaReferencePreset(); },
      [this](bool preferMailbox) {
        swapchain.setPreferMailbox(preferMailbox);
        framebufferResized = true;
      },
      [this](int index) { requestSceneLoad(index); }};

  imguiLayer.record(cmd, renderPassManager.getImGuiRenderPass(),
                    swapchain.getExtent(), imageIndex, ui);
  if (imageIndex < renderResources.frameCount()) {
    renderResources.noteLayout(renderResources.frame(imageIndex).swapchainImage,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
}

void VulkanRenderer::createSynchronization() {
  size_t swapCount = swapchain.getImageCount();
  imageAvailable.resize(MAX_FRAMES_DRAWS);
  drawFences.resize(MAX_FRAMES_DRAWS);
  asyncComputeTimeline.resize(MAX_FRAMES_DRAWS, VK_NULL_HANDLE);
  asyncComputeTimelineValue.assign(MAX_FRAMES_DRAWS, 0);
  // renderFinished is per-swap-image, not per-frame-in-flight (see header
  // comment for the validation rationale).
  renderFinished.resize(swapCount);
  imagesInFlight.resize(swapCount, VK_NULL_HANDLE);
  frameImageInFlight.resize(MAX_FRAMES_DRAWS,
                            std::numeric_limits<uint32_t>::max());

  VkSemaphoreCreateInfo semInfo = {};
  semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphoreTypeCreateInfo timelineInfo = {};
  timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineInfo.initialValue = 0;
  VkSemaphoreCreateInfo timelineSemInfo = {};
  timelineSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  timelineSemInfo.pNext = &timelineInfo;
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkDevice dev = device.getLogicalDevice();
  for (size_t i = 0; i < MAX_FRAMES_DRAWS; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable[i]) !=
            VK_SUCCESS ||
        vkCreateSemaphore(dev, &timelineSemInfo, nullptr,
                          &asyncComputeTimeline[i]) != VK_SUCCESS ||
        vkCreateFence(dev, &fenceInfo, nullptr, &drawFences[i]) != VK_SUCCESS)
      throw std::runtime_error("Failed to create synchronization primitives");
  }
  for (size_t i = 0; i < swapCount; i++) {
    if (vkCreateSemaphore(dev, &semInfo, nullptr, &renderFinished[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create renderFinished semaphore");
  }
}

void VulkanRenderer::createAsyncFrameCommandBuffers() {
  cleanupAsyncFrameCommandBuffers();

  const size_t count = swapchain.getImageCount();
  VkDevice dev = device.getLogicalDevice();
  shadowCommandBuffers.assign(count, VK_NULL_HANDLE);
  postCommandBuffers.assign(count, VK_NULL_HANDLE);
  ssaoCommandBuffers.assign(count, VK_NULL_HANDLE);

  auto allocate = [&](VkCommandPool pool, std::vector<VkCommandBuffer> &out,
                      const char *label) {
    VkCommandBufferAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(out.size());
    if (!out.empty() && vkAllocateCommandBuffers(dev, &ai, out.data()) !=
                            VK_SUCCESS) {
      throw std::runtime_error(std::string("Failed to allocate ") + label +
                               " command buffers");
    }
  };

  allocate(device.getGraphicsCommandPool(), shadowCommandBuffers, "shadow");
  allocate(device.getGraphicsCommandPool(), postCommandBuffers, "post");
  allocate(device.getComputeCommandPool(), ssaoCommandBuffers, "SSAO compute");
}

void VulkanRenderer::cleanupAsyncFrameCommandBuffers() {
  VkDevice dev = device.getLogicalDevice();
  if (dev == VK_NULL_HANDLE)
    return;

  auto freeBuffers = [&](VkCommandPool pool, std::vector<VkCommandBuffer> &bufs) {
    if (!bufs.empty()) {
      vkFreeCommandBuffers(dev, pool, static_cast<uint32_t>(bufs.size()),
                           bufs.data());
      bufs.clear();
    }
  };

  freeBuffers(device.getGraphicsCommandPool(), shadowCommandBuffers);
  freeBuffers(device.getGraphicsCommandPool(), postCommandBuffers);
  freeBuffers(device.getComputeCommandPool(), ssaoCommandBuffers);
}

void VulkanRenderer::createThreadedCommandResources() {
  cleanupThreadedCommandResources();

  const uint32_t hardwareThreads =
      std::max(1u, std::thread::hardware_concurrency());
  // Keep this conservative: command recording benefits from a few workers, but
  // oversubscribing the CPU can hurt frame pacing more than it helps.
  threadedCommandWorkerCount =
      std::min<uint32_t>(4u, std::max(1u, hardwareThreads > 1
                                              ? hardwareThreads - 1u
                                              : 1u));
  commandThreadPool =
      std::make_unique<CommandThreadPool>(threadedCommandWorkerCount);

  QueueFamilyIndices qi = device.getQueueFamilies();
  VkDevice dev = device.getLogicalDevice();
  threadedCommandFrames.assign(MAX_FRAMES_DRAWS, {});

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                   VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = static_cast<uint32_t>(qi.graphicsFamily);

  for (size_t frame = 0; frame < threadedCommandFrames.size(); ++frame) {
    threadedCommandFrames[frame].resize(threadedCommandWorkerCount);
    for (uint32_t worker = 0; worker < threadedCommandWorkerCount; ++worker) {
      ThreadCommandFrameResources &res = threadedCommandFrames[frame][worker];
      if (vkCreateCommandPool(dev, &poolInfo, nullptr, &res.pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create worker command pool");

      VkCommandBufferAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = res.pool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
      allocInfo.commandBufferCount = 1;
      if (vkAllocateCommandBuffers(dev, &allocInfo, &res.gBufferSecondary) !=
          VK_SUCCESS)
        throw std::runtime_error("Failed to allocate G-buffer secondary command buffer");
    }
  }

  spdlog::info("Phase 7.4 threaded command recording: {} worker(s)",
               threadedCommandWorkerCount);
}

void VulkanRenderer::cleanupThreadedCommandResources() {
  commandThreadPool.reset();

  VkDevice dev = device.getLogicalDevice();
  if (dev != VK_NULL_HANDLE) {
    for (auto &frameResources : threadedCommandFrames) {
      for (ThreadCommandFrameResources &res : frameResources) {
        if (res.pool != VK_NULL_HANDLE) {
          vkDestroyCommandPool(dev, res.pool, nullptr);
          res.pool = VK_NULL_HANDLE;
          res.gBufferSecondary = VK_NULL_HANDLE;
        }
      }
    }
  }

  threadedCommandFrames.clear();
  threadedCommandWorkerCount = 0;
}
