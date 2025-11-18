#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

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

struct PointLight {
        vec3 position;
        float intensity;
        vec3 color;
        float range;
};

layout (binding = 2, std430) buffer PointLights {
        PointLight gpu_point_lights[];
};

struct SpotLight {
        vec3 position;
        float intensity;
        vec3 direction;
        float inner_cosine;
        vec3 color;
        float outer_cosine;
};

layout (binding = 3, std430) buffer SpotLights {
        SpotLight gpu_spot_lights[];
};

vec3 evaluate_light(vec3 light_direction, vec3 light_color, vec3 normal, vec3 view_direction) {
        float diffuse_strength = max(dot(normal, light_direction), 0.0f);
        vec3 diffuse = diffuse_strength * light_color * albedo_color;

        vec3 half_vector = normalize(light_direction + view_direction);
        float specular_strength = pow(max(dot(normal, half_vector), 0.0f), max(shininess, 1.0f));
        vec3 specular = specular_strength * light_color * specular_color;

        return diffuse + specular;
}

void main() {
        vec3 normal = normalize(f_normal);
        vec3 view_direction = normalize(camera_position - f_position);

        vec3 color = ambient_color * ambient_intensity * albedo_color;

        vec3 dir_light_direction = normalize(-directional_direction);
        vec3 dir_radiance = directional_color * directional_intensity;
        color += evaluate_light(dir_light_direction, dir_radiance, normal, view_direction);

        for (uint i = 0; i < point_light_count; ++i) {
                PointLight light = gpu_point_lights[i];
                vec3 light_vector = light.position - f_position;
                float distance = length(light_vector);
                float attenuation = light.intensity / max(distance * distance, 1e-4f);
                float range_factor = clamp(1.0f - distance / max(light.range, 0.001f), 0.0f, 1.0f);
                attenuation *= range_factor;
                vec3 radiance = light.color * attenuation;
                vec3 light_direction = normalize(light_vector);
                color += evaluate_light(light_direction, radiance, normal, view_direction);
        }

        for (uint i = 0; i < spot_light_count; ++i) {
                SpotLight light = gpu_spot_lights[i];
                vec3 light_vector = light.position - f_position;
                float distance = length(light_vector);
                float attenuation = light.intensity / max(distance * distance, 1e-4f);
                vec3 light_direction = normalize(light_vector);
                vec3 fragment_direction = -light_direction;
                float theta = dot(light.direction, fragment_direction);
                float epsilon = max(light.inner_cosine - light.outer_cosine, 0.0001f);
                float smooth_factor = clamp((theta - light.outer_cosine) / epsilon, 0.0f, 1.0f);
                attenuation *= smooth_factor;
                vec3 radiance = light.color * attenuation;
                color += evaluate_light(light_direction, radiance, normal, view_direction);
        }

        final_color = vec4(color, 1.0f);
}
