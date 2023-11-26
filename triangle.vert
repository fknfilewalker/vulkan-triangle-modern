#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

layout(push_constant, std430) uniform pushConstant
{
    uint64_t vertexPtr;
};
layout(buffer_reference, std430) readonly buffer Vertex
{
    vec4 position;
};

void main() {
	Vertex vertex = Vertex(vertexPtr + gl_VertexIndex);
	gl_Position = vec4(vertex.position.xy, 0.0, 1.0);
}