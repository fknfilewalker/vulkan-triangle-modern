/* Usage: 
slangc shaders.slang -target spirv -emit-spirv-directly -fvk-use-entrypoint-name -source-embed-style u32 -source-embed-name shaders_spv -o shaders.h
*/
[[vk::push_constant]] float3* vertices;

[shader("vertex")]
float4 vertexMain(uint vid : SV_VertexID) : SV_Position
{
    return float4(vertices[vid], 1.0);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}