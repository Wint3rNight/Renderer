#pragma once


// Keep only the general utility structures here
struct QueueFamilyIndices {
  int graphicsFamily = -1;

  // check if the queue family indices are valid
  bool isValid() { return graphicsFamily >= 0; }
};
