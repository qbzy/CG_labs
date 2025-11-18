#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec3 camera_position; float _pad0;
        vec3 ambient_color; float ambient_intensity;
        vec3 directional_direction; float directional_intensity;
        vec3 directional_color; float _pad1;
        uint point_light_count;
        uint spot_light_count;
        vec2 _pad2;
};

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        mat4 normal_matrix;
        vec3 albedo_color; float shininess;
        vec3 specular_color; float _pad3;
};

void main() {
        vec4 world_position = model * vec4(v_position, 1.0f);
        vec4 world_normal = normal_matrix * vec4(v_normal, 0.0f);

        gl_Position = view_projection * world_position;

        f_position = world_position.xyz;
        f_normal = normalize(world_normal.xyz);
        f_uv = v_uv;
}
