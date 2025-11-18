#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 4;
constexpr uint32_t max_spot_lights = 4;

struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;

};

struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 camera_position; float _pad0;
        veekay::vec3 ambient_color; float ambient_intensity;
        veekay::vec3 directional_direction; float directional_intensity;
        veekay::vec3 directional_color; float _pad1;
        uint32_t point_light_count;
        uint32_t spot_light_count;
        veekay::vec2 _pad2;
};

struct ModelUniforms {
        veekay::mat4 model;
        veekay::mat4 normal_matrix;
        veekay::vec3 albedo_color; float shininess;
        veekay::vec3 specular_color; float _pad0;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        // NOTE: Model matrix (translation, rotation and scaling)
        veekay::mat4 matrix() const;

        // NOTE: Normal matrix for transforming normals without translation
        veekay::mat4 normal_matrix() const;
};

struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
        float shininess = 32.0f;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

        // NOTE: View matrix of camera (inverse of a transform)
        veekay::mat4 view() const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio) const;

        // NOTE: Forward direction derived from camera yaw/pitch rotation (in degrees)
        veekay::vec3 forward() const;
};

struct AmbientLight {
        veekay::vec3 color = {0.2f, 0.2f, 0.2f};
        float intensity = 0.2f;
};

struct DirectionalLight {
        veekay::vec3 direction = {0.3f, -1.0f, 0.2f};
        veekay::vec3 color = {1.0f, 1.0f, 1.0f};
        float intensity = 1.0f;
};

struct PointLight {
        veekay::vec3 position = {};
        float intensity = 10.0f;
        veekay::vec3 color = {1.0f, 0.9f, 0.8f};
        float range = 10.0f;
};

struct SpotLight {
        veekay::vec3 position = {};
        veekay::vec3 direction = {0.0f, -1.0f, 0.0f};
        veekay::vec3 color = {1.0f, 1.0f, 1.0f};
        float intensity = 25.0f;
        float inner_cutoff_degrees = 12.5f;
        float outer_cutoff_degrees = 17.5f;
};

struct GPUPointLight {
        veekay::vec3 position;
        float intensity;
        veekay::vec3 color;
        float range;
};

struct GPUSpotLight {
        veekay::vec3 position;
        float intensity;
        veekay::vec3 direction;
        float inner_cosine;
        veekay::vec3 color;
        float outer_cosine;
};

// NOTE: Scene objects
inline namespace {
        Camera camera{
                .position = {0.0f, -0.5f, -3.0f}
        };

        std::vector<Model> models;

        AmbientLight ambient_light{};
        DirectionalLight directional_light{};
        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_buffer;
        veekay::graphics::Buffer* spot_lights_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 composeRotationMatrix(const veekay::vec3& rotation_degrees) {
        const float pitch = toRadians(rotation_degrees.x);
        const float yaw = toRadians(rotation_degrees.y);
        const float roll = toRadians(rotation_degrees.z);

        const veekay::mat4 rotation_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, pitch);
        const veekay::mat4 rotation_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, yaw);
        const veekay::mat4 rotation_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, roll);

        // NOTE: Apply ZYX order (roll -> yaw -> pitch)
        return rotation_z * rotation_y * rotation_x;
}

veekay::mat4 Transform::matrix() const {
        const veekay::mat4 translation_matrix = veekay::mat4::translation(position);
        const veekay::mat4 rotation_matrix = composeRotationMatrix(rotation);
        const veekay::mat4 scale_matrix = veekay::mat4::scaling(scale);

        return translation_matrix * rotation_matrix * scale_matrix;
}

veekay::mat4 Transform::normal_matrix() const {
        constexpr float min_scale = 1e-4f;
        veekay::vec3 safe_scale = scale;
        safe_scale.x = std::max(safe_scale.x, min_scale);
        safe_scale.y = std::max(safe_scale.y, min_scale);
        safe_scale.z = std::max(safe_scale.z, min_scale);

        const veekay::vec3 inverse_scale = {
                1.0f / safe_scale.x,
                1.0f / safe_scale.y,
                1.0f / safe_scale.z
        };

        const veekay::mat4 rotation_matrix = composeRotationMatrix(rotation);
        const veekay::mat4 inverse_scale_matrix = veekay::mat4::scaling(inverse_scale);

        return rotation_matrix * inverse_scale_matrix;
}

veekay::vec3 Camera::forward() const {
        const float pitch = toRadians(rotation.x);
        const float yaw = toRadians(rotation.y);

        veekay::vec3 forward_vector{
                std::sin(yaw) * std::cos(pitch),
                std::sin(pitch),
                std::cos(yaw) * std::cos(pitch)
        };

        return veekay::vec3::normalized(forward_vector);
}

veekay::mat4 Camera::view() const {
        const veekay::vec3 forward_vector = forward();
        const veekay::vec3 world_up = {0.0f, -1.0f, 0.0f};

        const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward_vector, world_up));
        const veekay::vec3 up = veekay::vec3::cross(right, forward_vector);

        veekay::mat4 view_matrix = veekay::mat4::identity();

        view_matrix[0][0] = right.x;
        view_matrix[1][0] = right.y;
        view_matrix[2][0] = right.z;

        view_matrix[0][1] = up.x;
        view_matrix[1][1] = up.y;
        view_matrix[2][1] = up.z;

        view_matrix[0][2] = -forward_vector.x;
        view_matrix[1][2] = -forward_vector.y;
        view_matrix[2][2] = -forward_vector.z;

        view_matrix[3][0] = -veekay::vec3::dot(right, position);
        view_matrix[3][1] = -veekay::vec3::dot(up, position);
        view_matrix[3][2] = veekay::vec3::dot(forward_vector, position);

        return view_matrix;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

        return projection * view();
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        { // NOTE: Build graphics pipeline
                vertex_shader_module = loadShaderModule("shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

                fragment_shader_module = loadShaderModule("shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

                {
                        VkDescriptorPoolSize pools[] = {
                                {
                                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        .descriptorCount = 8,
                                },
                                {
                                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                        .descriptorCount = 8,
                                },
                                {
                                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        .descriptorCount = 8,
                                },
                                {
                                        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        .descriptorCount = 8,
                                }
                        };
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
                {
                        VkDescriptorSetLayoutBinding bindings[] = {
                                {
                                        .binding = 0,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                                {
                                        .binding = 1,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                                {
                                        .binding = 2,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                                {
                                        .binding = 3,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                        };

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(GPUPointLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        spot_lights_buffer = new veekay::graphics::Buffer(
                max_spot_lights * sizeof(GPUSpotLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
                VkDescriptorBufferInfo buffer_infos[] = {
                        {
                                .buffer = scene_uniforms_buffer->buffer,
                                .offset = 0,
                                .range = sizeof(SceneUniforms),
                        },
                        {
                                .buffer = model_uniforms_buffer->buffer,
                                .offset = 0,
                                .range = sizeof(ModelUniforms),
                        },
                        {
                                .buffer = point_lights_buffer->buffer,
                                .offset = 0,
                                .range = max_point_lights * sizeof(GPUPointLight),
                        },
                        {
                                .buffer = spot_lights_buffer->buffer,
                                .offset = 0,
                                .range = max_spot_lights * sizeof(GPUSpotLight),
                        },
                };

                VkWriteDescriptorSet write_infos[] = {
                        {
                                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = descriptor_set,
                                .dstBinding = 0,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .pBufferInfo = &buffer_infos[0],
                        },
                        {
                                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = descriptor_set,
                                .dstBinding = 1,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                .pBufferInfo = &buffer_infos[1],
                        },
                        {
                                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = descriptor_set,
                                .dstBinding = 2,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .pBufferInfo = &buffer_infos[2],
                        },
                        {
                                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = descriptor_set,
                                .dstBinding = 3,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .pBufferInfo = &buffer_infos[3],
                        },
                };

                vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

        // NOTE: Add models to scene
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
                .specular_color = veekay::vec3{0.04f, 0.04f, 0.04f},
                .shininess = 8.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {-2.0f, -0.5f, -1.5f},
                },
                .albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
                .specular_color = veekay::vec3{1.0f, 0.5f, 0.5f},
                .shininess = 32.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {1.5f, -0.5f, -0.5f},
                },
                .albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
                .specular_color = veekay::vec3{0.5f, 1.0f, 0.5f},
                .shininess = 48.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {0.0f, -0.5f, 1.0f},
                },
                .albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f},
                .specular_color = veekay::vec3{0.5f, 0.5f, 1.0f},
                .shininess = 64.0f
        });

        point_lights = {
                PointLight{
                        .position = {-1.0f, -0.1f, 0.0f},
                        .intensity = 30.0f,
                        .color = {1.0f, 0.85f, 0.7f},
                        .range = 8.0f,
                }
        };

        spot_lights = {
                SpotLight{
                        .position = {1.5f, -0.25f, -1.0f},
                        .direction = veekay::vec3{0.0f, -0.2f, 1.0f},
                        .color = {0.7f, 0.8f, 1.0f},
                        .intensity = 40.0f,
                        .inner_cutoff_degrees = 15.0f,
                        .outer_cutoff_degrees = 22.0f,
                }
        };
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_buffer;
        delete spot_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
        // NOTE: Build an always-visible UI with runtime controls for the scene
        ImGui::Begin("Controls:", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextWrapped("Hold the left mouse button to orbit the camera and use WASDQZ for movement.");

        ImGui::Separator();
        ImGui::TextUnformatted("Ambient light");
        ImGui::ColorEdit3("Color##Ambient", ambient_light.color.elements);
        ImGui::SliderFloat("Intensity##Ambient", &ambient_light.intensity, 0.0f, 5.0f, "%.2f");

        ImGui::Separator();
        ImGui::TextUnformatted("Directional light");
        ImGui::ColorEdit3("Color##Directional", directional_light.color.elements);
        ImGui::SliderFloat3("Direction##Directional", directional_light.direction.elements, -1.0f, 1.0f);
        ImGui::SliderFloat("Intensity##Directional", &directional_light.intensity, 0.0f, 10.0f, "%.2f");

        if (!point_lights.empty()) {
                // NOTE: Expose point light parameters at runtime
                ImGui::Separator();
                ImGui::TextUnformatted("Point lights");
                for (size_t i = 0; i < point_lights.size(); ++i) {
                        ImGui::PushID(static_cast<int>(i));
                        ImGui::Text("Point #%zu", i + 1);
                        ImGui::DragFloat3("Position", point_lights[i].position.elements, 0.05f, -10.0f, 10.0f);
                        ImGui::ColorEdit3("Color", point_lights[i].color.elements);
                        ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 100.0f, "%.1f");
                        ImGui::SliderFloat("Range", &point_lights[i].range, 0.1f, 20.0f, "%.1f");
                        ImGui::PopID();
                }
        }

        if (!spot_lights.empty()) {
                // NOTE: Expose spot light parameters with smooth cutoff tweaking
                ImGui::Separator();
                ImGui::TextUnformatted("Spot lights");
                for (size_t i = 0; i < spot_lights.size(); ++i) {
                        ImGui::PushID(static_cast<int>(100 + i));
                        ImGui::Text("Spot #%zu", i + 1);
                        ImGui::DragFloat3("Position", spot_lights[i].position.elements, 0.05f, -10.0f, 10.0f);
                        ImGui::DragFloat3("Direction", spot_lights[i].direction.elements, 0.02f, -1.0f, 1.0f);
                        ImGui::ColorEdit3("Color", spot_lights[i].color.elements);
                        ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 100.0f, "%.1f");
                        ImGui::SliderFloat("Inner cutoff", &spot_lights[i].inner_cutoff_degrees, 1.0f, 89.0f, "%.1f");
                        ImGui::SliderFloat("Outer cutoff", &spot_lights[i].outer_cutoff_degrees, 1.0f, 90.0f, "%.1f");
                        spot_lights[i].outer_cutoff_degrees = std::max(spot_lights[i].outer_cutoff_degrees, spot_lights[i].inner_cutoff_degrees + 0.5f);
                        ImGui::PopID();
                }
        }

        ImGui::End();

        const bool interacting_with_ui = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || ImGui::IsAnyItemActive();

        if (!interacting_with_ui) {
                // NOTE: Only process camera controls when the cursor is not over the UI
                using namespace veekay::input;

                const veekay::vec3 world_up = {0.0f, -1.0f, 0.0f};
                const veekay::vec3 front = camera.forward();
                const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
                const veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

                if (mouse::isButtonDown(mouse::Button::left)) {
                        const auto move_delta = mouse::cursorDelta();
                        constexpr float look_sensitivity = 0.1f;
                        camera.rotation.x = std::clamp(camera.rotation.x - move_delta.y * look_sensitivity, -89.0f, 89.0f);
                        camera.rotation.y -= move_delta.x * look_sensitivity;
                }

                constexpr float move_speed = 0.1f;
                if (keyboard::isKeyDown(keyboard::Key::w))
                        camera.position += front * move_speed;

                if (keyboard::isKeyDown(keyboard::Key::s))
                        camera.position -= front * move_speed;

                if (keyboard::isKeyDown(keyboard::Key::d))
                        camera.position += right * move_speed;

                if (keyboard::isKeyDown(keyboard::Key::a))
                        camera.position -= right * move_speed;

                if (keyboard::isKeyDown(keyboard::Key::q))
                        camera.position += up * move_speed;

                if (keyboard::isKeyDown(keyboard::Key::z))
                        camera.position -= up * move_speed;
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        // NOTE: Normalize the user-provided direction to avoid undefined math in shaders
        veekay::vec3 directional_dir = directional_light.direction;
        const float directional_length = veekay::vec3::length(directional_dir);
        if (directional_length > 1e-4f) {
                directional_dir /= directional_length;
        } else {
                directional_dir = {0.0f, -1.0f, 0.0f};
        }

        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .camera_position = camera.position,
                .ambient_color = ambient_light.color,
                .ambient_intensity = ambient_light.intensity,
                .directional_direction = directional_dir,
                .directional_intensity = directional_light.intensity,
                .directional_color = directional_light.color,
                ._pad2 = {0.0f, 0.0f},
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
                const Model& model = models[i];
                ModelUniforms& uniforms = model_uniforms[i];

                uniforms.model = model.transform.matrix();
                uniforms.normal_matrix = model.transform.normal_matrix();
                uniforms.albedo_color = model.albedo_color;
                uniforms.specular_color = model.specular_color;
                uniforms.shininess = model.shininess;
                uniforms._pad0 = 0.0f;
        }

        const uint32_t point_light_count = std::min<uint32_t>(static_cast<uint32_t>(point_lights.size()), max_point_lights);
        const uint32_t spot_light_count = std::min<uint32_t>(static_cast<uint32_t>(spot_lights.size()), max_spot_lights);
        scene_uniforms.point_light_count = point_light_count;
        scene_uniforms.spot_light_count = spot_light_count;

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

        // NOTE: Upload point light data into the storage buffer (unused slots stay zeroed out)
        std::memset(point_lights_buffer->mapped_region, 0, max_point_lights * sizeof(GPUPointLight));
        auto* gpu_point_lights = static_cast<GPUPointLight*>(point_lights_buffer->mapped_region);
        for (uint32_t i = 0; i < point_light_count; ++i) {
                        gpu_point_lights[i] = GPUPointLight{
                                .position = point_lights[i].position,
                                .intensity = point_lights[i].intensity,
                                .color = point_lights[i].color,
                                .range = point_lights[i].range,
                        };
        }

        // NOTE: Upload spot light data with precomputed cosine cutoffs for the shader
        std::memset(spot_lights_buffer->mapped_region, 0, max_spot_lights * sizeof(GPUSpotLight));
        auto* gpu_spot_lights = static_cast<GPUSpotLight*>(spot_lights_buffer->mapped_region);
        for (uint32_t i = 0; i < spot_light_count; ++i) {
                const SpotLight& spot = spot_lights[i];
                veekay::vec3 normalized_direction = spot.direction;
                const float direction_length = veekay::vec3::length(normalized_direction);
                if (direction_length > 1e-4f) {
                        normalized_direction /= direction_length;
                } else {
                        normalized_direction = {0.0f, -1.0f, 0.0f};
                }
                const float inner_cos = std::cos(toRadians(std::min(spot.inner_cutoff_degrees, spot.outer_cutoff_degrees - 0.01f)));
                const float outer_cos = std::cos(toRadians(spot.outer_cutoff_degrees));

                gpu_spot_lights[i] = GPUSpotLight{
                        .position = spot.position,
                        .intensity = spot.intensity,
                        .direction = normalized_direction,
                        .inner_cosine = inner_cos,
                        .color = spot.color,
                        .outer_cosine = outer_cos,
                };
        }

        const size_t alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
