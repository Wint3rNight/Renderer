#include "InputManager.h"

void InputManager::init(GLFWwindow *newWindow) {
  window = newWindow;

  // Get initial window size for mouse centering
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  lastX = static_cast<float>(width) / 2.0f;
  lastY = static_cast<float>(height) / 2.0f;

  // Store 'this' in the window user pointer so static callbacks can access it
  glfwSetWindowUserPointer(window, this);

  // Register callbacks
  glfwSetCursorPosCallback(window, mouseCallback);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

  // Capture the cursor for FPS-style camera.
  // Raw mouse motion avoids XGrabPointer on X11, which can suppress touchpad
  // events while keyboard keys are held simultaneously.
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  if (glfwRawMouseMotionSupported())
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

  // Sticky keys: a press+release that both happen within one (long) frame —
  // e.g. tapping a key during a scene-switch device-idle wait — still reads
  // as GLFW_PRESS on the next poll instead of being silently dropped.
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
}

void InputManager::pollEvents() {
  // Reset per-frame deltas before polling
  deltaX = 0.0f;
  deltaY = 0.0f;
  glfwPollEvents();
}

bool InputManager::isKeyPressed(int key) const {
  return glfwGetKey(window, key) == GLFW_PRESS;
}

bool InputManager::wasKeyJustPressed(int key) {
  bool down = isKeyPressed(key);
  bool &prev = prevKeyDown[key];
  bool justPressed = down && !prev;
  prev = down;
  return justPressed;
}

glm::vec2 InputManager::getMouseDelta() const { return {deltaX, deltaY}; }

glm::vec2 InputManager::getMousePosition() const { return {lastX, lastY}; }

void InputManager::resetMouseDelta() {
  deltaX = 0.0f;
  deltaY = 0.0f;
}

bool InputManager::shouldClose() const { return glfwWindowShouldClose(window); }

bool InputManager::wasResized() const { return framebufferResized; }

void InputManager::resetResizedFlag() { framebufferResized = false; }

// --- Static GLFW callbacks ---

void InputManager::mouseCallback(GLFWwindow *w, double xposIn, double yposIn) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  if (!self)
    return;

  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (self->firstMouse) {
    self->lastX = xpos;
    self->lastY = ypos;
    self->firstMouse = false;
  }

  self->deltaX += xpos - self->lastX;
  self->deltaY += self->lastY - ypos; // reversed: y goes bottom to top

  self->lastX = xpos;
  self->lastY = ypos;
}

void InputManager::framebufferResizeCallback(GLFWwindow *w, int /*width*/,
                                             int /*height*/) {
  auto *self = static_cast<InputManager *>(glfwGetWindowUserPointer(w));
  if (!self)
    return;
  self->framebufferResized = true;
}
