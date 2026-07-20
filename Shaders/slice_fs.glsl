#version 460 core

// Inputs
// none

// Uniforms
layout (std140) uniform CameraMatrices {
    mat4 view_matrix;
    mat4 perspective_matrix;
};
uniform mat4 model_matrix;

// TODO: define const dimensions, halo, cell_size in a header
uniform float cell_size;
uniform uvec3 face_dimensions;
uniform uvec3 halo;

uniform vec2 resolution;

uniform float plain_offset;

uniform vec4 core_color;
uniform sampler3D sample_textures[6];

// Outputs
out vec4 color;

// facing order from EulerFluidSim:
// Z+, X+, Y+, Z-, X-, Y-
uint GetFaceIndex(vec3 v) {
   vec3 abs_v = abs(v);
   float m = max(max(abs_v.x, abs_v.y), abs_v.z);
   return uint(v.x == m) +
          2 * uint(v.y == m) +
          3 * uint(v.z == -m) +
          4 * uint(v.x == -m) +
          5 * uint(v.y == -m);
}

// For consistency axes have to be passed, but for performance it is much cheaper to hard-code them here
vec3 ConvertToLocal(uint face_index, vec3 v) {
    return float(face_index == 0) * v +
           float(face_index == 1) * v.yzx +
           float(face_index == 2) * v.zxy +
           float(face_index == 3) * (-v) +
           float(face_index == 4) * (-v.yzx) +
           float(face_index == 5) * (-v.zxy);
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

    // x and y are equals here
    float inner_sphere_radius = float(face_dimensions.x) * cell_size * 0.5f;
    float outer_sphere_radius = inner_sphere_radius + float(face_dimensions.z) * cell_size;

    if (length(model_coords) > outer_sphere_radius) {
        discard;
    } else if (length(model_coords) < inner_sphere_radius) {
        color = core_color;
        return;
    }

    // transform into appropriate grid coordinates
    uint face_index = GetFaceIndex(model_coords);
    vec3 face_coords = ConvertToLocal(face_index, model_coords);
    vec3 base_grid_coords = normalize(face_coords) * inner_sphere_radius;
    vec3 regular_grid_coords = vec3(base_grid_coords.x, base_grid_coords.y, length(face_coords) - inner_sphere_radius);
    float adjusted_radius = inner_sphere_radius + halo.x * cell_size;
    // bring to [0,1] range
    // texture includes halo cells, so actual range has to be smaller
    // regular_grid_coords.xy is in range: [-adjusted_radius; adjusted_radius]
    // regular_grid_coords.z is in range: [1; face_dimensions.z + halo.z * 2]
    vec3 uv_coords = (regular_grid_coords + vec3(adjusted_radius, adjusted_radius, 1.0f)) /
                     vec3(adjusted_radius * 2.0f, adjusted_radius * 2.0f, face_dimensions.z + float(halo.z * 2));

    color = texture(sample_textures[face_index], uv_coords);

    vec4 clip = perspective_matrix * view_matrix * vec4(plain_intersection, 1.0);
    float z_ndc = clip.z / clip.w;
    gl_FragDepth = 0.5 * z_ndc + 0.5;
}
