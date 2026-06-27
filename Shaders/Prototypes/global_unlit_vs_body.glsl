// definition options:
// 1. USE_VERTEX_COLOR - is color vertex attribute used?
// 2. UVS_COUNT - how much uvs to pass, if any, X is [0,8]

#define COLOR_ATTRIB_POS UVS_COUNT + 1

// Inputs setup
layout(position=0) in vec3 vertex_position;
#if UVS_COUNT > 0
    layout(position=1) in vec2 vertex_uvs[UVS_COUNT];
#endif
#if defined(USE_VERTEX_COLOR)
    layout(position=COLOR_ATTRIB_POS) in vec3 vertex_color;
#endif


// Uniforms
layout (std140) uniform CameraMatrices {
    mat4 view_matrix;
    mat4 perspective_matrix;
};
uniform mat4 model_matrix;

// Outputs
#if UVS_COUNT > 0
    out vec2 uvs[UVS_COUNT];
#endif
#if defined(USE_VERTEX_COLOR)
    out vec4 color;
#endif

void main() {
#if UVS_COUNT > 0
    for(int i = 0; i < UVS_COUNT; ++i) {
        uvs[i] = vertex_uvs[i];
    }
#endif
#if defined(USE_VERTEX_COLOR)
    color = vec4(vertex_color, 1.0f);
#endif

    gl_Position = perspective_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f);
}