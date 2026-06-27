#version 460 core

// Inputs
// none

// Uniforms
layout (std140) uniform CameraMatrices {
    mat4 view_matrix;
    mat4 perspective_matrix;
};
uniform mat4 model_matrix;
uniform vec3 slice_bounds;
uniform vec2 resolution;

uniform float plain_offset;

uniform sampler3D sample_texture;

// Outputs
out vec4 color;

void main() {
    mat4 inverse_perspective = inverse(perspective_matrix);
    mat4 inverse_view = inverse(view_matrix);
    mat4 inverse_model = inverse(model_matrix);

    vec3 camera_world_pos = inverse_view[3].xyz;
    vec3 plain_normal = normalize(camera_world_pos);

    vec2 ndc = (gl_FragCoord.xy / resolution) * 2.0f - 1.0f;
    vec4 ray_near = inverse_perspective * vec4(ndc, -1.0f, 1.0f);
    vec4 ray_far = inverse_perspective * vec4(ndc, 1.0f, 1.0f);
    ray_near /= ray_near.w;
    ray_far /= ray_far.w;

    vec3 ray_start_world = (inverse_view * ray_near).xyz;
    vec3 ray_end_world = (inverse_view * ray_far).xyz;
    vec3 ray_direction = normalize(ray_end_world - ray_start_world);

    vec3 plain_intersection =
        ray_start_world +
        dot(plain_normal, plain_normal * plain_offset - ray_start_world) / dot(plain_normal, ray_direction) *
        ray_direction;

    vec3 model_coords = (inverse_model * vec4(plain_intersection, 1.0f)).xyz;

    if (model_coords.x > slice_bounds.x ||
        model_coords.x < -slice_bounds.x ||
        model_coords.y > slice_bounds.y ||
        model_coords.y < -slice_bounds.y ||
        model_coords.z > slice_bounds.z ||
        model_coords.z < -slice_bounds.z) {
        discard;
    }

    // bring to [0,1] range
    vec3 uv_coords = (model_coords + slice_bounds) / slice_bounds * 0.5f;

    color = texture(sample_texture, uv_coords);

    vec4 clip = perspective_matrix * view_matrix * vec4(plain_intersection, 1.0);
    float z_ndc = clip.z / clip.w;
    gl_FragDepth = 0.5 * z_ndc + 0.5;
}
