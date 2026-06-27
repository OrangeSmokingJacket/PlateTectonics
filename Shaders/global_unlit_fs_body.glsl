// definition options:
// 1. USE_VERTEX_COLOR
// 2. USE_TEXTURE
// 3. USE_ALPHA_CLIPPING

// Inputs
#if defined(USE_VERTEX_COLOR)
    in vec4 v_color;
#endif
#if defined(USE_TEXTURE)
    in vec2 v_uvs;
#endif

// Uniforms
uniform vec4 base_color;
#if defined(USE_TEXTURE)
    uniform sampler2D diffuse_texture;
    uniform vec2 uv_tiling;
    uniform vec2 uv_offset;
#endif
#if defined(USE_ALPHA_CLIPPING)
    uniform float alpha_threshold;
#endif

// Outputs
out vec4 color;

void main() {
    vec4 result_color = base_color;
#if defined(USE_VERTEX_COLOR)
    result_color *= v_color;
#endif

#if defined(USE_TEXTURE)
    vec2 computed_uv = v_uvs * uv_tiling + uv_offset;
    result_color *= texture(diffuse_texture, computed_uv);
#endif

#if defined(USE_ALPHA_CLIPPING)
    if (result_color.a < alpha_threshold) {
        discard;
    }
#endif

    color = result_color;
}