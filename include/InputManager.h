#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>

class InputManager {
public:
  InputManager() = default;
  ~InputManager() = default;

  void init(GLFWwindow *window);
  void pollEvents();

  // --- Key state ---
  bool isKeyPressed(int key) const;
  // Rising-edge detection: true exactly once per press. Tracks previous
  // state per key internally — call at most once per key per frame.
  bool wasKeyJustPressed(int key);

  // --- Mouse state ---
  glm::vec2 getMouseDelta() const;
  glm::vec2 getMousePosition() const;
  void resetMouseDelta();

  // --- Window state ---
  bool shouldClose() const;
  bool wasResized() const;
  void resetResizedFlag();

private:
  GLFWwindow *window = nullptr;

  // Mouse tracking
  float lastX = 0.0f;
  float lastY = 0.0f;
  float deltaX = 0.0f;
  float deltaY = 0.0f;
  bool firstMouse = true;

  // Resize flag
  bool framebufferResized = false;

  // Per-key previous state for wasKeyJustPressed edge detection
  std::unordered_map<int, bool> prevKeyDown;

  // Static callbacks (GLFW requires free/static functions)
  static void mouseCallback(GLFWwindow *w, double xpos, double ypos);
  static void framebufferResizeCallback(GLFWwindow *w, int width, int height);
};
