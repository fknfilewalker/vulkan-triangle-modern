#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

const uint64_t sizeOfFloat = 4ul;
const uint64_t sizeOfVec3 = 3ul * sizeOfFloat;

layout(buffer_reference, scalar) readonly buffer Vertex
{
    vec3 position;
};
layout(push_constant, scalar) uniform pushConstant
{
    uint64_t vertexPtr;
};

void main() {
	Vertex vertex = Vertex(vertexPtr + sizeOfVec3 * gl_VertexIndex);
	gl_Position = vec4(vertex.position.xy, 0.0, 1.0);
}