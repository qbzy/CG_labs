#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_shadow_position;

layout (location = 0) out vec4 final_color;

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

layout (binding = 2) uniform sampler2D textures[8];
layout (binding = 4) uniform sampler2DShadow shadow_texture;

void main() {
    vec3 base_color = albedo_color;
    if (use_texture > 0.5) {
        int texture_index = int(texture_info.x + 0.5);
        base_color = texture(textures[texture_index], f_uv).rgb;
    }
    
    // Нормализуем нормаль
    vec3 normal = normalize(f_normal);
    
    // Направление к наблюдателю (view direction)
    vec3 view_dir = normalize(view_position - f_position);
    
    // Половинчатый вектор (half vector) для модели Блинна-Фонга
    // h = normalize(view_dir - sun_light_direction)
    // (sun_light_direction указывает К источнику, поэтому используем вычитание)
    vec3 half_vector = normalize(view_dir - sun_light_direction);
    
    // n·l - скалярное произведение нормали и направления света
    // sun_light_direction указывает К источнику, поэтому используем отрицание
    float n_dot_l = max(-dot(sun_light_direction, normal), 0.0);
    
    // Диффузная составляющая: ρd(n·l)
    vec3 diffuse = base_color * n_dot_l;
    
    // Спекулярная составляющая: ρs(n·h)^n
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    vec3 specular = specular_color * pow(n_dot_h, shininess);
    
    // Формула Блинна-Фонга: [ρd(n·l) + ρs(n·h)^n](n·l)EL(l)
    // где EL(l) = sun_light_color
    vec3 sun_light_intensity = n_dot_l * sun_light_color * (diffuse + specular);
    
    // Вычисление теней
    vec3 proj_coords = f_shadow_position.xyz / f_shadow_position.w;
    // Transform to [0, 1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    
    float shadow_factor = 1.0;
    
    // Perform shadow lookup
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || 
        proj_coords.y < 0.0 || proj_coords.y > 1.0 || 
        proj_coords.z < 0.0 || proj_coords.z > 1.0) {
        // Outside light frustum
        shadow_factor = 1.0;
    } else {
        // Адаптивный bias на основе угла падения света
        // Когда свет падает под малым углом (почти параллельно поверхности),
        // возникает проблема с точностью глубины - нужен больший bias
        vec3 light_dir = normalize(-sun_light_direction);
        float cos_angle = max(dot(normal, light_dir), 0.0);
        
        // Вычисляем угол между нормалью и направлением света
        // Чем меньше cos_angle (больше угол), тем больше bias нужен
        // Используем квадрат для более агрессивного увеличения bias при малых углах
        float angle_factor = 1.0 - cos_angle;
        float bias = 0.0001 + 0.003 * angle_factor * angle_factor;
        
        // Ограничиваем bias разумными пределами
        bias = min(bias, 0.01);
        
        vec3 biased_coords = vec3(proj_coords.xy, proj_coords.z - bias);
        
        // Sample shadow map
        shadow_factor = texture(shadow_texture, biased_coords);
    }
    
    // Итоговый цвет: рассеянный свет + направленный свет с учетом теней
    vec3 color = ambient_light_intensity * base_color + sun_light_intensity * shadow_factor;
    
    final_color = vec4(color, 1.0);
}
