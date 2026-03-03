#version 450

layout(location = 0) in vec3 fragColor; // interpolated from vertex shader

layout(location = 0) out vec4 outColor; // final color output

void main() {
  outColor = vec4(fragColor, 1.0); // set alpha to 1.0 (fully opaque)
}
