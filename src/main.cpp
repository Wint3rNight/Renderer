#include "Camera.h"
#include "InputManager.h"
#include "VulkanRenderer.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;
InputManager inputManager;

// Camera settings — positioned inside Sponza hall, looking along the length
Camera camera(glm::vec3(0.0f, 2.0f, 0.0f));

// Timing
float deltaTime = 0.0f;
// double, not float: glfwGetTime() is seconds since init, and float's ~2 ms
// resolution past 4.5 hours of uptime would quantize deltaTime into visible
// camera/animation stutter. The delta itself is small and safe as float.
double lastFrame = 0.0;

void processInput() {
  if (inputManager.isKeyPressed(GLFW_KEY_ESCAPE))
    glfwSetWindowShouldClose(window, true);

  if (inputManager.isKeyPressed(GLFW_KEY_W))
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_S))
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_A))
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_D))
    camera.ProcessKeyboard(RIGHT, deltaTime);
}

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // Enable window resizing

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }

  // InputManager handles cursor callback and resize callback
  inputManager.init(window);
}

int main(int argc, char **argv) {
  initWindow("Vulkan Renderer", 1920, 1080);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  camera.MovementSpeed = 15.0f;
  vulkanRenderer.setCameraPresetCallback(
      [](glm::vec3 position, float yaw, float pitch, float speed) {
        camera.SetPose(position, yaw, pitch);
        camera.MovementSpeed = speed;
      });
  bool cameraMode = true;  // Tab toggles between camera fly-through and ImGui
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  // Scenes are data-driven: Resources/Scenes/*.scene files describe models,
  // transforms, animations, instanced rings, and the camera pose. The ImGui
  // Camera panel exposes a scene picker for runtime switching. An optional
  // CLI argument selects the startup scene: an index into the discovered
  // list, or a path to a .scene file.
  vulkanRenderer.discoverScenes("Resources/Scenes");
  bool sceneLoaded = false;
  if (argc > 1) {
    const std::string arg = argv[1];
    try {
      if (!arg.empty() &&
          arg.find_first_not_of("0123456789") == std::string::npos) {
        // An out-of-range index must NOT count as loaded, or the fallback
        // below is skipped and the app runs with an empty scene.
        sceneLoaded = vulkanRenderer.loadSceneAt(std::stoi(arg));
      } else {
        vulkanRenderer.loadScene(SceneIO::loadFromFile(arg));
        sceneLoaded = true;
      }
    } catch (const std::exception &e) {
      std::cerr << "Failed to load requested scene '" << arg
                << "': " << e.what() << std::endl;
    }
  }
  if (!sceneLoaded && vulkanRenderer.hasScenes()) {
    vulkanRenderer.loadSceneAt(0);
  } else if (!sceneLoaded) {
    // No scene files found — fall back to the classic built-in setup.
    SceneDescription fallback;
    fallback.name = "Sponza (built-in fallback)";
    SceneModelEntry sponza;
    sponza.path = "Resources/Models/Sponza/glTF/Sponza.gltf";
    fallback.models.push_back(sponza);
    SceneRingEntry ring;
    ring.path = "Resources/Models/DamagedHelmet/glTF/DamagedHelmet.gltf";
    ring.count = 64;
    fallback.rings.push_back(ring);
    vulkanRenderer.loadScene(fallback);
  }

  while (!inputManager.shouldClose()) {
    double currentFrame = glfwGetTime();
    deltaTime = static_cast<float>(currentFrame - lastFrame);
    lastFrame = currentFrame;

    inputManager.pollEvents();
    processInput();

    // Tab toggles between camera fly-through (cursor hidden) and UI mode
    // (cursor visible, ImGui panels interactive).
    // WantCaptureMouse can't drive this because GLFW_CURSOR_DISABLED delivers
    // only raw deltas — ImGui never sees absolute positions and can't detect
    // hover.
    if (inputManager.wasKeyJustPressed(GLFW_KEY_TAB)) {
      cameraMode = !cameraMode;
      glfwSetInputMode(window, GLFW_CURSOR,
                       cameraMode ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
      // Entering UI mode with the HUD hidden would strand the user in a
      // dead-looking state (free cursor, nothing to click, no mouse-look) —
      // un-hide the panels so UI mode always shows a UI.
      if (!cameraMode && !vulkanRenderer.isUiVisible())
        vulkanRenderer.toggleUiVisibility();
    }

    // F1 toggles HUD-free screenshot mode (hides all ImGui panels).
    if (inputManager.wasKeyJustPressed(GLFW_KEY_F1))
      vulkanRenderer.toggleUiVisibility();

    // Mouse drives camera only in camera mode
    glm::vec2 mouseDelta = inputManager.getMouseDelta();
    if (cameraMode && (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f)) {
      camera.ProcessMouseMovement(mouseDelta.x, mouseDelta.y);
    }

    // Handle resize
    if (inputManager.wasResized()) {
      vulkanRenderer.notifyResize();
      inputManager.resetResizedFlag();
    }

    // Advances node animations (dynamic objects) and global transforms.
    vulkanRenderer.updateScene(currentFrame);

    vulkanRenderer.updateCameraView(camera.GetViewMatrix(), camera.Position);

    // Feed ImGui camera state; read back speed in case the slider changed it
    vulkanRenderer.setImGuiCameraInfo(camera.Position, camera.MovementSpeed);
    vulkanRenderer.draw();
    camera.MovementSpeed = vulkanRenderer.getCameraSpeed();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
