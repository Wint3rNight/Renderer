#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "AutoExposurePass.h"
#include "BloomPass.h"
#include "CompositePass.h"
#include "DescriptorManager.h"
#include "GpuDrivenGBufferPass.h"
#include "ImGuiLayer.h"
#include "MaterialProbePass.h"
#include "Model.h"
#include "ModelManager.h"
#include "PerformanceMetrics.h"
#include "RenderResources.h"
#include "RenderPassManager.h"
#include "Scene.h"
#include "SceneNode.h"
#include "ShadowPass.h"
#include "SsaoPass.h"
#include "SsgiPass.h"
#include "TextureManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"
#include "VulkanPipelineCache.h"
#include "VulkanSwapchain.h"

struct InstancedDrawable {
  int modelId;
  std::vector<InstanceData> instances;
  AllocatedBuffer instanceBuffer;
  // World-space bounding sphere covering all instances. Used for whole-batch
  // frustum culling and LOD selection.
  glm::vec3 groupCenter = glm::vec3(0.0f);
  float groupRadius = 0.0f;
};

class CommandThreadPool;

class VulkanRenderer {
public:
  VulkanRenderer();

  using CameraPresetCallback =
      std::function<void(glm::vec3 position, float yaw, float pitch,
                         float speed)>;

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);

  // --- Scene management ---
  // Scans `directory` for *.scene files (also checked under ../ because the
  // binary runs from build/). Call once after init(), before loadSceneAt().
  void discoverScenes(const std::string &directory = "Resources/Scenes");
  bool hasScenes() const { return !availableScenes.empty(); }
  // Immediate load: waits for the device to go idle, clears the scene graph
  // and instanced drawables, then builds the described scene. Models are
  // cached by path, so returning to a scene reuses its GPU geometry.
  void loadScene(const SceneDescription &desc);
  // Returns false (and leaves the current scene untouched) when the index
  // is out of range — callers with a fallback must check this.
  bool loadSceneAt(int index);
  // Deferred load, safe to call from UI callbacks that run mid-record; the
  // switch happens at the top of the next draw().
  void requestSceneLoad(int index);
  // Advances node animations and recomputes global transforms. Call once per
  // frame before draw() with the elapsed time in seconds.
  void updateScene(float timeSeconds);

  SceneNode &getRootNode();
  void updateCameraView(const glm::mat4 &viewMatrix,
                        const glm::vec3 &cameraPosition);
  void addInstancedModel(int modelId, const std::vector<glm::mat4> &transforms);
  void cleanup();
  void notifyResize();

  void setImGuiCameraInfo(glm::vec3 pos, float speed);
  void setFov(float fovDegrees);
  void setDrawDistance(float dist);
  void setCameraPresetCallback(CameraPresetCallback callback);
  float getCameraSpeed() const { return imguiCameraSpeed; }
  bool imguiWantsMouse() const { return imguiLayer.wantsMouse(); }
  void toggleUiVisibility() { imguiLayer.toggleVisibility(); }
  bool isUiVisible() const { return imguiLayer.isVisible(); }

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;
  int currentFrame = 0;

  SceneNode rootNode;
  SceneUniformBuffer sceneUbo = {};

  // --- Scene library ---
  std::vector<SceneDescription> availableScenes;
  std::vector<std::string> availableSceneNames; // display names for ImGui
  int activeSceneIndex = -1;
  int pendingSceneIndex = -1;   // set by the UI, applied at next draw()
  std::string activeEnvironmentHdr; // HDRI backing the current sky/IBL
  VkFormat shadowDepthFormat = VK_FORMAT_UNDEFINED;

  // --- G-buffer (per swapchain image) ---
  VkFormat gBufferDepthFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> gBuffer0Images, gBuffer1Images, gBuffer2Images,
      gBufferDepthImages;
  std::vector<ImageViewHandle> gBuffer0Views, gBuffer1Views, gBuffer2Views,
      gBufferDepthViews;
  std::vector<VkFramebuffer> gBufferFramebuffers;

  // --- Lit buffer (post-PBR HDR, sampled by SSR composite pass) ---
  VkFormat litFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> litImages;
  std::vector<ImageViewHandle> litViews;
  std::vector<VkFramebuffer> litFramebuffers;
  VkSampler litSampler = VK_NULL_HANDLE;

  // --- TAA history (HDR, persistent across frames) ---
  // Same ring sizing as SSGI: MAX_FRAMES_DRAWS + 1 avoids overwriting a
  // history image that a still-in-flight frame is sampling.
  std::vector<AllocatedImage> taaHistoryImages;   // size MAX_FRAMES_DRAWS + 1
  std::vector<ImageViewHandle> taaHistoryViews;   // size MAX_FRAMES_DRAWS + 1
  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that drift just past the edge fall
  // back gracefully (treated as off-screen by the shader's bounds check).
  VkSampler taaSampler = VK_NULL_HANDLE;
  bool autoExpEnabled = true;   // ImGui toggle; off -> manual EV only.

  // --- IBL resources ---
  int iblSkyboxImageIndex = -1; // index into TextureManager::textureImages
  int irradianceImageIndex = -1;
  int prefilteredEnvImageIndex = -1;
  int brdfLutImageIndex = -1;
  int ssaoNoiseImageIndex = -1;
  VkSampler iblSampler = VK_NULL_HANDLE;
  VkSampler ssaoNoiseSampler = VK_NULL_HANDLE;

  // --- Instanced drawables ---
  std::vector<InstancedDrawable> instancedDrawables;

  // --- GPU-driven rendering controls (Phase 7.6) ---
  bool imguiGpuDrivenEnabled = true;
  bool imguiHzbCullingEnabled = true;
  int imguiGpuDrivenMinCandidates = 256;
  bool imguiThreadedGBufferEnabled = false;

  // --- Multi-threaded command recording (Phase 7.4) ---
  struct ThreadCommandFrameResources {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer gBufferSecondary = VK_NULL_HANDLE;
  };
  std::vector<std::vector<ThreadCommandFrameResources>> threadedCommandFrames;
  std::unique_ptr<CommandThreadPool> commandThreadPool;
  uint32_t threadedCommandWorkerCount = 0;

  // --- Subsystems ---
  VulkanDevice device;
  VulkanSwapchain swapchain;
  RenderPassManager renderPassManager;
  DescriptorManager descriptorManager;
  VulkanPipeline pipeline;
  VulkanPipelineCache pipelineCache;
  TextureManager textureManager;
  ModelManager modelManager;
  PerformanceMetrics metrics;
  RenderResources renderResources;
  GpuDrivenGBufferPass gpuDrivenGBufferPass;
  MaterialProbePass materialProbePass;
  BloomPass bloomPass;
  AutoExposurePass autoExposurePass;
  ShadowPass shadowPass;
  SsaoPass ssaoPass;
  SsgiPass ssgiPass;
  CompositePass compositePass;

  // --- Synchronization ---
  // imageAvailable & drawFences are sized by MAX_FRAMES_DRAWS (frames in
  // flight). renderFinished is sized by swapchain image count and indexed
  // by the acquired imageIndex — the presentation engine binds its wait
  // to the swap image, not to our frame-in-flight slot. Indexing this by
  // currentFrame caused a validation hit when the swap engine had not
  // finished presenting the image whose semaphore we tried to reuse.
  // imagesInFlight maps swap image -> frame fence. frameImageInFlight is the
  // reverse owner map, used to clear stale image entries when a frame fence is
  // reused for a different acquired image.
  // See https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkSemaphore> asyncComputeTimeline;
  std::vector<uint64_t> asyncComputeTimelineValue;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;
  std::vector<uint32_t> frameImageInFlight;

  // Split graphics/compute command buffers for Phase 7.5 async SSAO.
  // swapchain.getCommandBuffer(image) records the G-buffer submit; these
  // additional primaries record shadow work and post-lighting work.
  std::vector<VkCommandBuffer> shadowCommandBuffers;
  std::vector<VkCommandBuffer> postCommandBuffers;
  std::vector<VkCommandBuffer> ssaoCommandBuffers;
  bool framebufferResized = false;

  // --- ImGui ---
  ImGuiLayer imguiLayer;
  CameraPresetCallback cameraPresetCallback;
  glm::vec3 imguiCameraPos = {};
  float imguiCameraSpeed = 15.0f;
  float imguiCameraFov = 45.0f;
  // Metric world since node-transform baking: Sponza is ~30 m long, so a
  // 200 m far plane covers everything with depth precision to spare.
  float imguiDrawDistance = 200.0f;
  float imguiLodNear = 15.0f;
  float imguiLodFar = 45.0f;
  float imguiPointShadowFar = 40.0f;
  // Fog tunables (mirrored into sceneUbo.fogParams so the shader can read).
  float imguiFogDensity = 0.0f;
  float imguiFogClamp = 0.0f;
  int imguiDebugMode = 0;
  float imguiSpecAAVariance = 1.60f;       // tunable
  // Karis-style spec-AA threshold cap. 0.18 (the plan's KAPPA) leaves
  // visible per-pixel sparkle on Sponza's dense cloth weave / foliage at
  // glancing angles. Bumping to 0.5 lets the kernel saturate more, folding
  // sub-pixel normal variance into much higher effective roughness on
  // high-frequency surfaces while leaving low-variance surfaces unchanged.
  float imguiSpecAAThreshold = 1.0f;       // tunable
  // IBL specular mip floor. This only affects global cubemap specular, not
  // direct-light highlights. Keeping it moderately high prevents the
  // environment map's bright texels from reading as rectangular local lights
  // on glossy indoor floors.
  float imguiIblRoughnessFloor = 0.65f;    // tunable
  bool imguiShadowFrontFaceCull = false;   // off → CULL_NONE in shadow pass.
                                           // Front-face cull silently drops
                                           // any single-sided wall whose only
                                           // triangle faces the sun, so the
                                           // floor's shadow lookup misses the
                                           // wall as an occluder and reads as
                                           // lit. NONE captures both winding
                                           // orientations; slope-scaled bias
                                           // + receiver-plane gradient
                                           // already handle the acne that
                                           // front-cull was protecting.
  bool imguiCullShadowCasters = false;      // opt-in only: tight CSM caster
                                           // culling can remove off-slice
                                           // occluders that still cast into
                                           // the cascade.
  float imguiCsmFar = 80.0f;               // cascade far plane (metric world)
  float imguiIblIntensity = 1.0f;          // manual IBL/sky multiplier
  // Exposure stops applied BEFORE the ACES tonemap in second.frag. ACES has
  // a steep toe — without a pre-scale, midtones get crushed to black on
  // shadowed Sponza interiors. +1 to +2 stops typically matches the warm
  // reference look once the sky-occlusion proxy is also lifted.
  float imguiExposureEV = 0.0f;            // exposure stops, -3..+3
  // Min-roughness floor applied at the G-buffer write for non-metallic
  // surfaces only. Sponza glTF's roughness maps bottom out too low — that
  // produces the wet-floor banner reflection and stone-pillar glitter.
  // 0 = disabled (use authored roughness); ~0.35–0.5 is the Sponza sweet
  // spot. Metals (metallic > 0.5) are untouched so trim still reads glossy.
  float imguiMinSurfaceRoughness = 0.60f;
  // Sky-occlusion proxy floor — bottom of the IBL multiplier in shadowed
  // pixels. 0.30 = old conservative ("dead-black interior"), 0.55 = new
  // default ("warm interior"), 1.0 = unbounded (full sky everywhere).
  float imguiSkyOcclusionFloor = 0.55f;
  // SSGI (screen-space one-bounce diffuse) intensity. 0 = disabled.
  // Dropped 1.0 → 0.6 alongside the per-sample firefly clamp tightening
  // and the close-range fade: 1.0 over-emphasized the residual SSGI grain
  // on sun-lit floor pixels once auto-exposure started amplifying. 0.6
  // keeps the warm-bounce feel without the visible noise floor.
  float imguiSsgiIntensity = 0.6f;
  // SSGI sample quality. 8 is the default performance/quality balance; 12 is
  // the old high-quality path, useful for A/B checks when chasing artifacts.
  int imguiSsgiSamples = 8;
  bool imguiEnableSunDirect = true;
  bool imguiEnablePointLights = true;
  bool imguiEnableSpotLights = true;
  bool imguiEnableIblAmbient = true;
  bool imguiEnableSsgiBounce = true;
  bool imguiEnableBloom = true;
  float imguiBloomThreshold = 0.8f;
  float imguiBloomIntensity = 0.12f;
  float imguiBloomRadius = 1.25f;
  bool imguiSsrEnabled = false;            // off by default: avoids rough Sponza SSR leaks
  bool imguiTaaEnabled = false;
  bool imguiResponsiveTaa = true;          // drop history while camera moves
  bool imguiPreferMailboxPresent = true;   // on = MAILBOX / uncapped present
  // CAS-style sharpening strength. Dropped 0.4 → 0.10 — the previous 0.4
  // was amplifying the residual SSGI/spec-AA noise into a visible chunky
  // pattern on the floor. 0.10 still restores TAA-softening on real edges
  // without exaggerating noise. Push back toward 0.3–0.4 only if the
  // image looks too soft after the noise-reduction pass.
  float imguiSharpness = 0.10f;
  float imguiEdgeAA = 0.35f;              // post/TAA edge cleanup
  float imguiAlphaDither = 0.45f;         // alpha-test cutout stabilization
  float imguiShadowSoftness = 1.25f;      // PCF radius multiplier
  float imguiContactGrounding = 0.45f;    // SSAO contact-strength multiplier
  bool imguiDayNightEnable = false;        // day/night animation
  float imguiDayNightSpeed = 60.0f;        // sim-hours per real-second
  float imguiDayNightHour = 12.0f;         // current sim-hour [0..24)
  float imguiNormalStrength = 0.55f;       // Sponza normals are over-sharp at 1.0
  bool imguiUseGeomNormalOnly = false;     // P2 diag: bypass normal-map sampling
  MaterialProbeResult materialProbeResult = {};
  glm::uvec2 materialProbePixel = glm::uvec2(0);

  void applySponzaReferencePreset();
  void updateMaterialProbePixel();

  // --- TAA state ---
  // Frame counter drives the Halton index and temporal history ring slots.
  // Jitter wraps every 8 frames; history slots wrap by their ring size.
  uint32_t taaFrameCounter = 0;
  // Un-jittered baseline projection. Updated only by rebuildProjection().
  // Each draw() starts by copying this into sceneUbo.projection BEFORE
  // applying the per-frame Halton jitter, so the jitter doesn't accumulate
  // across frames into a random-walk drift of the projection matrix.
  glm::mat4 taaBaseProjection = glm::mat4(1.0f);
  // Jittered VP of the previous frame. Used by the TAA shader to
  // reproject each pixel's world position to where it landed in last
  // frame's image (which was rendered with last frame's jitter).
  glm::mat4 taaPrevViewProj = glm::mat4(1.0f);
  bool taaHistoryValid = false;            // dropped on swapchain recreate / first frame
  glm::vec3 taaLastCameraPos = glm::vec3(0.0f);
  glm::mat4 taaLastView = glm::mat4(1.0f);
  bool taaHasLastCamera = false;
  bool cameraMovedThisFrame = false;

  // --- Init helpers ---
  void createSynchronization();
  void createAsyncFrameCommandBuffers();
  void cleanupAsyncFrameCommandBuffers();
  void createThreadedCommandResources();
  void cleanupThreadedCommandResources();
  void registerSwapchainResources();
  void noteGBufferFinalLayouts(uint32_t imageIndex);
  void noteLitFinalLayout(uint32_t imageIndex);
  void noteCompositeFinalLayouts(uint32_t imageIndex, uint32_t historyIndex);
  void createGBuffer();
  void cleanupGBuffer();
  void createLitResources();
  void createLitFramebuffers();
  void cleanupLitFramebuffers();
  void cleanupLitResources();
  void createTaaResources();
  void cleanupTaaResources();
  void initIBL();
  void cleanupIBL();
  void rebuildProjection();

  // --- Per-frame ---
  void recordCommands(uint32_t currentImage);
  void recordGBufferCommands(uint32_t currentImage);
  void recordShadowCommands(VkCommandBuffer cmd);
  void recordSsaoComputeCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordPostCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                         const VkViewport &viewport,
                         const VkRect2D &scissor);
  void recordTransparentPass(VkCommandBuffer cmd, uint32_t currentImage,
                             const VkViewport &viewport,
                             const VkRect2D &scissor);
  void recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex);
  void recreateSwapChain();
};
