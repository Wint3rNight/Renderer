#version 450

layout(location = 0) in vec3 fragCol; // color from vertex shader
layout(location = 0) out vec4 outColor; // final color output

void main() {
  outColor = vec4(fragCol, 1.0);
}
