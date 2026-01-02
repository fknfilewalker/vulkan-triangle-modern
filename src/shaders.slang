/* Usage:
slangc shaders.slang -target spirv -fvk-invert-y -fvk-use-entrypoint-name -source-embed-style u32 -source-embed-name shaders_spv -o shaders.h
*/
[vk::push_constant] float4* vertices;

struct OutIn
{
    float4 position : SV_POSITION, color : COLOR;
};

[shader("vertex")]
OutIn vertexMain(uint vid : SV_VertexID)
{
    return { vertices[vid], { float3(uint3(0, 1, 2) == (vid % 3)), 1.0 } };
}

[shader("fragment")]
float4 fragmentMain(OutIn input) : SV_Target
{
    return input.color;
}
