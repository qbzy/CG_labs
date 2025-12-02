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

layout (set = 0, binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
    DirectionalLight directional_light;
    vec3 camera_position;
    float _pad0;
};

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float use_albedo_texture;
    float use_specular_texture;
    float use_emissive_texture;
    float emissive_strength;
};

layout (set = 0, binding = 2, std430) readonly buffer PointLightsSSBO {
    PointLight point_lights[];
};

layout (set = 1, binding = 0) uniform sampler2D albedo_map;
layout (set = 1, binding = 1) uniform sampler2D specular_map;
layout (set = 1, binding = 2) uniform sampler2D emissive_map;

vec3 calculate_directional_light(DirectionalLight light, vec3 normal, vec3 view_dir, vec3 base_color, vec3 specular_factor) {
    vec3 light_dir = normalize(-light.direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.color * diff * base_color;

    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = light.color * spec * specular_factor;

    return (diffuse + specular) * light.intensity;
}

vec3 calculate_point_light(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir, vec3 base_color, vec3 specular_factor) {
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
    vec3 specular = light.color * spec * specular_factor;

    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (1 + light.linear * distance + light.quadratic * (distance * distance));

    return (diffuse + specular) * light.intensity * attenuation * spot_factor;
}

vec3 sample_warped_color(sampler2D tex) {
    vec2 warp = f_uv * 2.5 + vec2(sin(f_position_world.x * 1.2 + f_uv.y * 8.0),
                                  cos(f_position_world.z * 1.1 + f_uv.x * 6.0)) * 0.18;
    vec3 base = texture(tex, warp).rgb;
    vec3 offset = texture(tex, warp + vec2(0.15 * sin(warp.y * 6.0), 0.12 * cos(warp.x * 4.0))).rgb;
    return mix(base, offset, 0.35);
}

vec3 sample_specular_map() {
    vec2 spiral = f_uv * 3.0 + vec2(cos(f_position_world.x * 0.7), sin(f_position_world.z * 0.5)) * 0.1;
    vec3 main_spec = texture(specular_map, spiral).rgb;
    vec3 detail = texture(specular_map, spiral * 0.5 + vec2(0.2, -0.1)).rgb;
    return clamp(main_spec * 0.7 + detail * 0.3, 0.0, 1.0);
}

vec3 sample_emissive() {
    vec2 pulse = f_uv * vec2(1.0, -1.0) + vec2(sin(f_position_world.y + f_uv.x * 10.0),
                                              cos(f_position_world.y + f_uv.y * 12.0)) * 0.05;
    vec3 glow = texture(emissive_map, pulse).rgb;
    vec3 blur = texture(emissive_map, pulse + vec2(0.2, 0.0)).rgb;
    return mix(glow, blur, 0.5);
}

void main() {
   vec3 norm = normalize(f_normal_world);
   vec3 view_dir = normalize(camera_position - f_position_world);

   vec3 sampled_albedo = sample_warped_color(albedo_map);
   vec3 sampled_specular = sample_specular_map();
   vec3 emissive = sample_emissive() * emissive_strength * use_emissive_texture;

   vec3 base_color = mix(albedo_color, sampled_albedo, use_albedo_texture);
   vec3 specular_factor = mix(specular_color, sampled_specular, use_specular_texture);

   float ambient_strength = 0.1f;
   vec3 ambient = ambient_strength * base_color;

   vec3 result = ambient + calculate_directional_light(directional_light, norm, view_dir, base_color, specular_factor);

   for (int i = 0; i < point_lights.length(); ++i) {
       result += calculate_point_light(point_lights[i], norm, f_position_world, view_dir, base_color, specular_factor);
   }

   result += emissive;
   final_color = vec4(result, 1.0);
}
