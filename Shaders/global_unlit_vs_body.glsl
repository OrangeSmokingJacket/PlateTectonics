// definition options:
// 1. USE_VERTEX_COLOR
// 2. USE_TEXTURE

#if defined(USE_VERTEX_COLOR)
    #define UV_POS 2
#else
    #define UV_POS 1
#endif

// Inputs
layout(location=0) in vec3 vertex_position;
#if defined(USE_VERTEX_COLOR)
    layout(location=1) in vec4 vertex_color;
#endif
#if defined(USE_TEXTURE)
    layout(location=UV_POS) in vec2 vertex_uvs;
#endif

// Uniforms
layout (std140) uniform CameraMatrices {
    mat4 view_matrix;
    mat4 perspective_matrix;
};
uniform mat4 model_matrix;

// Outputs
#if defined(USE_TEXTURE)
    out vec2 v_uvs;
#endif
#if defined(USE_VERTEX_COLOR)
    out vec4 v_color;
#endif

void main() {
#if defined(USE_TEXTURE)
    v_uvs = vertex_uvs;
#endif
#if defined(USE_VERTEX_COLOR)
    v_color = vertex_color;
#endif

    gl_Position = perspective_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f);
}
