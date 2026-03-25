#include "VulkanRenderer.h"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
}

int main() {
  initWindow("Vulkan Renderer", 800, 600);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  float angle = 0.0f;
  float deltaTime = 0.0f;
  float lastTime = 0.0f;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    float now = glfwGetTime();
    deltaTime = now - lastTime;
    lastTime = now;

    angle += 10.0f * deltaTime;
    if (angle > 360.0f) {
      angle -= 360.0f;
    }

    glm::mat4 firstModel(1.0f);
    glm::mat4 secondModel(1.0f);

    firstModel = glm::translate(firstModel, glm::vec3(-2.0f, 0.0f, -5.0f));
    firstModel = glm::rotate(firstModel, glm::radians(angle),
                             glm::vec3(0.0f, 0.0f, 1.0f));
    secondModel = glm::translate(secondModel, glm::vec3(2.0f, 0.0f, -5.0f));
    secondModel = glm::rotate(secondModel, glm::radians(-angle * 100.0f),
                              glm::vec3(0.0f, 0.0f, 1.0f));

    vulkanRenderer.updateModel(0, firstModel);
    vulkanRenderer.updateModel(1, secondModel);

    vulkanRenderer.draw();
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
