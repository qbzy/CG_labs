#version 450

layout (location = 0) in vec3 f_position_world;
layout (location = 1) in vec3 f_normal_world;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct DirectionalLight {
    vec3 direction;
    float _pad0;
    vec3 color;
    float intensity;
};

struct PointLight {
    vec3 position;
    float type;
    vec3 direction;
    float cutOff;
    vec3 color;
    float intensity;
    float constant;
    float linear;
    float quadratic;
    float outerCutOff;
};

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
    DirectionalLight directional_light;
    vec3 camera_position;
    float _pad0;
};

layout (push_constant) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float use_texture;
    int texture_index;
    float _pad[2];
};

layout (binding = 2, std430) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

vec3 calculate_directional_light(DirectionalLight light, vec3 normal, vec3 view_dir, vec3 base_color) {
    vec3 light_dir = normalize(-light.direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.color * diff * base_color;

    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = light.color * spec * specular_color;

    return (diffuse + specular) * light.intensity;
}

vec3 calculate_point_light(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir, vec3 base_color) {
    if (light.intensity <= 0.0) {
        return vec3(0.0);
    }

    vec3 light_dir = normalize(light.position - frag_pos);

    float spot_factor = 1.0;
    if (light.type > 0.5) {
        float theta = dot(light_dir, normalize(-light.direction));
        float epsilon = light.cutOff - light.outerCutOff;
        spot_factor = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    }

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.color * diff * base_color;

    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = light.color * spec * specular_color;

    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (1 + light.linear * distance + light.quadratic * (distance * distance));

    return (diffuse + specular) * light.intensity * attenuation * spot_factor;
}

void main() {
   vec3 norm = normalize(f_normal_world);
   vec3 view_dir = normalize(camera_position - f_position_world);

   vec3 base_color = albedo_color;

   float ambient_strength = 0.1f;
   vec3 ambient = ambient_strength * base_color;

   vec3 result = ambient + calculate_directional_light(directional_light, norm, view_dir, base_color);

   for (int i = 0; i < point_lights.length(); ++i) {
       result += calculate_point_light(point_lights[i], norm, f_position_world, view_dir, base_color);
   }

   final_color = vec4(result, 1.0);
}