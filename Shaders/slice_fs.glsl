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

uniform vec4 core_color;
uniform sampler3D sample_textures[6];

// Outputs
out vec4 color;

// facing order from EulerFluidSim:
// Z+, X+, Y+, Z-, X-, Y-
int GetFaceIndex(vec3 v) {
   vec3 abs_v = abs(v);
   float m = max(max(abs_v.x, abs_v.y), abs_v.z);
   return int(v.x == m) +
          2 * int(v.y == m) +
          3 * int(v.z == -m) +
          4 * int(v.x == -m) +
          5 * int(v.y == -m);
}

// For consistency axes have to be passed, but for performance it is much cheaper to hard-code them here
vec3 ConvertToLocal(int face_index, vec3 v) {
    return int(face_index == 0) * v +
           int(face_index == 1) * v.yzx +
           int(face_index == 2) * v.zxy +
           int(face_index == 3) * (-v) +
           int(face_index == 4) * (-v.yzx) +
           int(face_index == 5) * (-v.zxy);
}

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

    // slice_bounds is already half the sim size
    // x and y are equals here
    float inner_sphere_radius = slice_bounds.x / 2;
    float outer_sphere_radius = inner_sphere_radius + slice_bounds.z;

    if (length(model_coords) > outer_sphere_radius) {
        discard;
    } else if (length(model_coords) < inner_sphere_radius) {
        color = core_color;
        return;
    }

    // transform into appropriate grid coordinates
    int face_index = GetFaceIndex(model_coords);
    vec3 local_model_coords = ConvertToLocal(face_index, model_coords);
    vec3 base_lmc = (local_model_coords / local_model_coords.z) * inner_sphere_radius;
    vec3 regular_grid_coords = vec3(base_lmc.x, base_lmc.y, length(local_model_coords) - inner_sphere_radius);
    // bring to [0,1] range
    vec3 uv_coords = vec3(regular_grid_coords.x + inner_sphere_radius,
                          regular_grid_coords.y + inner_sphere_radius,
                          regular_grid_coords.z) / slice_bounds;

    color = texture(sample_textures[face_index], uv_coords);

    vec4 clip = perspective_matrix * view_matrix * vec4(plain_intersection, 1.0);
    float z_ndc = clip.z / clip.w;
    gl_FragDepth = 0.5 * z_ndc + 0.5;
}
