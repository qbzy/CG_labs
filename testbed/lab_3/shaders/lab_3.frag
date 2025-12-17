#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float use_texture;
	int texture_index;
};

layout (binding = 2) uniform sampler2D textures[8];

void main() {
	vec3 base_color;
	if (use_texture > 0.5) {
		base_color = texture(textures[texture_index], f_uv).rgb;
	} else {
		base_color = albedo_color;
	}
	
	final_color = vec4(base_color, 1.0);
}
