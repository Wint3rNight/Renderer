#!/bin/bash


DIR=$(dirname "$0")
echo "Starting SPIR-V compilation in $DIR..."
VERT_SHADERS=$(find "$DIR" -maxdepth 1 -name "*.vert")
FRAG_SHADERS=$(find "$DIR" -maxdepth 1 -name "*.frag")

for shader in $VERT_SHADERS; do
    echo "Compiling $shader..."
    glslangValidator -V "$shader" -o "$shader.spv"
done

for shader in $FRAG_SHADERS; do
    echo "Compiling $shader..."
    glslangValidator -V "$shader" -o "$shader.spv"
done

echo "Done. Shaders compiled to .spv"

#chmod +x compile_shaders.sh then ./compile_shaders.sh
