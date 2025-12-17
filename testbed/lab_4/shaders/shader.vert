#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_shadow_position;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 shadow_projection;
    vec3 view_position;
    vec3 ambient_light_intensity;
    vec3 sun_light_direction;
    vec3 sun_light_color;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float use_texture;
    vec3 specular_color;
    float shininess;
    vec4 texture_info;
};

void main() {
    vec4 world_position = model * vec4(v_position, 1.0);
    f_position = world_position.xyz;
    
    // Нормальная матрица - только вращательная часть
    mat3 normal_matrix = mat3(transpose(inverse(model)));
    f_normal = normalize(normal_matrix * v_normal);
    
    f_shadow_position = shadow_projection * world_position;
    
    gl_Position = view_projection * world_position;
    f_uv = v_uv;
}
