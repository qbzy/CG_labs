#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
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
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
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
};

// NOTE: Scene objects
inline namespace {
        Camera camera{
                .position = {0.0f, 0.0f, -4.0f}
        };

        std::vector<Model> models;

        size_t cube_model_index = 0;
        size_t sphere_model_index = 0;
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

        Mesh cube_mesh;
        Mesh sphere_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture* texture;
        VkSampler texture_sampler;
}

namespace {

constexpr float rotation_sensitivity = 0.0025f;
constexpr float max_pitch = 1.5f;
constexpr float movement_speed = 4.0f;

constexpr float orbit_radius_min = 0.5f;
constexpr float orbit_radius_max = 5.0f;
constexpr float orbit_height_min = -2.0f;
constexpr float orbit_height_max = 2.0f;
constexpr float orbit_speed_min = -5.0f;
constexpr float orbit_speed_max = 5.0f;
constexpr float sphere_scale_min = 0.25f;
constexpr float sphere_scale_max = 1.5f;

float orbit_radius = 1.5f;
float orbit_height = 0.0f;
float orbit_speed = 1.0f;
float orbit_angle = 0.0f;
float sphere_scale = 0.5f;
bool orbit_animation_enabled = true;

} // namespace

float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
        auto translation = veekay::mat4::translation(position);
        auto scaling_matrix = veekay::mat4::scaling(scale);

        auto rotate_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
        auto rotate_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
        auto rotate_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

        auto rotation_matrix = rotate_y * rotate_x * rotate_z;

        return translation * rotation_matrix * scaling_matrix;
}

veekay::mat4 Camera::view() const {
        auto translation = veekay::mat4::translation(-position);

        auto rotate_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, -rotation.z);
        auto rotate_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, -rotation.x);
        auto rotate_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, -rotation.y);

        return translation * rotate_z * rotate_x * rotate_y;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
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
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
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
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
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

        // NOTE: Sphere mesh initialization
        {
                constexpr float radius = 0.5f;
                constexpr uint32_t slices = 32;
                constexpr uint32_t stacks = 16;

                std::vector<Vertex> vertices;
                vertices.reserve((stacks + 1) * (slices + 1));

                std::vector<uint32_t> indices;
                indices.reserve(stacks * slices * 6);

                for (uint32_t stack = 0; stack <= stacks; ++stack) {
                        float v = float(stack) / float(stacks);
                        float phi = v * float(M_PI);

                        float sin_phi = std::sin(phi);
                        float cos_phi = std::cos(phi);

                        for (uint32_t slice = 0; slice <= slices; ++slice) {
                                float u = float(slice) / float(slices);
                                float theta = u * 2.0f * float(M_PI);

                                float sin_theta = std::sin(theta);
                                float cos_theta = std::cos(theta);

                                veekay::vec3 normal{
                                        sin_phi * cos_theta,
                                        cos_phi,
                                        sin_phi * sin_theta,
                                };

                                veekay::vec3 position = normal * radius;

                                vertices.push_back(Vertex{
                                        position,
                                        veekay::vec3::normalized(normal),
                                        {u, v},
                                });
                        }
                }

                for (uint32_t stack = 0; stack < stacks; ++stack) {
                        for (uint32_t slice = 0; slice < slices; ++slice) {
                                uint32_t first = stack * (slices + 1) + slice;
                                uint32_t second = first + slices + 1;

                                indices.push_back(first);
                                indices.push_back(second);
                                indices.push_back(first + 1);

                                indices.push_back(first + 1);
                                indices.push_back(second);
                                indices.push_back(second + 1);
                        }
                }

                sphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
                        vertices.size() * sizeof(Vertex), vertices.data(),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

                sphere_mesh.index_buffer = new veekay::graphics::Buffer(
                        indices.size() * sizeof(uint32_t), indices.data(),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

                sphere_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Add models to scene
        models.clear();

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {0.0f, 0.0f, 0.0f},
                },
                .albedo_color = veekay::vec3{1.0f, 0.4f, 0.25f}
        });
        cube_model_index = models.size() - 1;

        models.emplace_back(Model{
                .mesh = sphere_mesh,
                .transform = Transform{
                        .position = {orbit_radius, orbit_height, 0.0f},
                        .scale = {sphere_scale, sphere_scale, sphere_scale},
                },
                .albedo_color = veekay::vec3{0.25f, 0.6f, 1.0f}
        });
        sphere_model_index = models.size() - 1;
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete sphere_mesh.index_buffer;
        delete sphere_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
        static bool first_frame = true;
        static double last_time = 0.0;

        if (first_frame) {
                last_time = time;
                first_frame = false;
        }

        float delta_time = static_cast<float>(time - last_time);
        last_time = time;

        if (delta_time < 0.0f)
                delta_time = 0.0f;

        ImGui::Begin("Controls");
        ImGui::SliderFloat("Field of view", &camera.fov, 30.0f, 120.0f);
        ImGui::SliderFloat("Orbit radius", &orbit_radius, orbit_radius_min, orbit_radius_max);
        ImGui::SliderFloat("Orbit height", &orbit_height, orbit_height_min, orbit_height_max);
        ImGui::SliderFloat("Orbit speed", &orbit_speed, orbit_speed_min, orbit_speed_max);
        ImGui::Checkbox("Animate orbit", &orbit_animation_enabled);
        if (!orbit_animation_enabled) {
                ImGui::SliderFloat("Orbit angle", &orbit_angle, 0.0f, 2.0f * float(M_PI));
        }
        ImGui::SliderFloat("Sphere scale", &sphere_scale, sphere_scale_min, sphere_scale_max);

        ImGui::Separator();
        ImGui::TextUnformatted("Orbit preview");

        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        float preview_height = 200.0f;
        canvas_size.x = std::max(canvas_size.x, 200.0f);
        canvas_size.y = preview_height;
        ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
        ImVec2 canvas_end{canvas_origin.x + canvas_size.x, canvas_origin.y + canvas_size.y};

        ImGui::InvisibleButton("##orbit_preview", canvas_size);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddRect(canvas_origin, canvas_end, IM_COL32(255, 255, 255, 64));

        ImVec2 canvas_center{canvas_origin.x + canvas_size.x * 0.5f,
                             canvas_origin.y + canvas_size.y * 0.5f};
        float orbit_visual_radius = 0.45f * std::min(canvas_size.x, canvas_size.y);

        draw_list->AddCircle(canvas_center, orbit_visual_radius, IM_COL32(180, 180, 180, 220), 96, 2.0f);
        draw_list->AddLine({canvas_center.x - orbit_visual_radius - 12.0f, canvas_center.y},
                           {canvas_center.x + orbit_visual_radius + 12.0f, canvas_center.y},
                           IM_COL32(255, 255, 255, 48), 1.0f);
        draw_list->AddLine({canvas_center.x, canvas_center.y - orbit_visual_radius - 12.0f},
                           {canvas_center.x, canvas_center.y + orbit_visual_radius + 12.0f},
                           IM_COL32(255, 255, 255, 48), 1.0f);
        draw_list->AddRectFilled({canvas_center.x - 5.0f, canvas_center.y - 5.0f},
                                 {canvas_center.x + 5.0f, canvas_center.y + 5.0f},
                                 IM_COL32(255, 102, 64, 255));

        ImVec2 sphere_screen_pos{canvas_center.x + std::cos(orbit_angle) * orbit_visual_radius,
                                 canvas_center.y + std::sin(orbit_angle) * orbit_visual_radius};
        draw_list->AddLine(canvas_center, sphere_screen_pos, IM_COL32(255, 255, 255, 96), 1.5f);
        draw_list->AddCircleFilled(sphere_screen_pos, 6.0f, IM_COL32(64, 153, 255, 255));

        char radius_label[64];
        std::snprintf(radius_label, sizeof(radius_label), "Radius: %.2f", orbit_radius);
        ImVec2 radius_text_size = ImGui::CalcTextSize(radius_label);
        ImVec2 radius_text_pos{canvas_end.x - radius_text_size.x - 6.0f,
                               canvas_end.y - radius_text_size.y - 6.0f};
        draw_list->AddText(radius_text_pos, IM_COL32(200, 220, 255, 200), radius_label);

        if (orbit_height_max > orbit_height_min) {
                float normalized_height = (orbit_height - orbit_height_min) /
                                          (orbit_height_max - orbit_height_min);
                normalized_height = std::clamp(normalized_height, 0.0f, 1.0f);
                float height_indicator_y = canvas_end.y - normalized_height * canvas_size.y;
                draw_list->AddLine({canvas_origin.x, height_indicator_y},
                                   {canvas_end.x, height_indicator_y}, IM_COL32(120, 200, 255, 120),
                                   1.0f);
                char height_label[64];
                std::snprintf(height_label, sizeof(height_label), "Height: %.2f", orbit_height);
                float label_y = height_indicator_y - 16.0f;
                label_y = std::clamp(label_y, canvas_origin.y + 4.0f, canvas_end.y - 20.0f);
                draw_list->AddText({canvas_origin.x + 6.0f, label_y}, IM_COL32(200, 220, 255, 200),
                                   height_label);
        }
        ImGui::End();

        const float two_pi = 2.0f * float(M_PI);

        if (orbit_animation_enabled) {
                orbit_angle += delta_time * orbit_speed;
        }

        if (orbit_angle > two_pi || orbit_angle < -two_pi) {
                orbit_angle = std::fmod(orbit_angle, two_pi);
        }

        auto& cube = models[cube_model_index];
        auto& sphere = models[sphere_model_index];

        sphere.transform.scale = {sphere_scale, sphere_scale, sphere_scale};

        float cos_angle = std::cos(orbit_angle);
        float sin_angle = std::sin(orbit_angle);

        sphere.transform.position = {
                cube.transform.position.x + cos_angle * orbit_radius,
                cube.transform.position.y + orbit_height,
                cube.transform.position.z + sin_angle * orbit_radius,
        };

        sphere.transform.rotation = {0.0f, -orbit_angle, 0.0f};

        const ImGuiIO& io = ImGui::GetIO();

        using namespace veekay::input;

        bool camera_control_active = !io.WantCaptureMouse && mouse::isButtonDown(mouse::Button::left);
        mouse::setCaptured(camera_control_active);

        if (camera_control_active) {
                auto move_delta = mouse::cursorDelta();

                camera.rotation.y += move_delta.x * rotation_sensitivity;
                camera.rotation.x += move_delta.y * rotation_sensitivity;

                if (camera.rotation.x > max_pitch)
                        camera.rotation.x = max_pitch;
                if (camera.rotation.x < -max_pitch)
                        camera.rotation.x = -max_pitch;

                if (camera.rotation.y > two_pi || camera.rotation.y < -two_pi)
                        camera.rotation.y = std::fmod(camera.rotation.y, two_pi);
        }

        auto view = camera.view();

        veekay::vec3 right{view[0][0], view[1][0], view[2][0]};
        veekay::vec3 up{view[0][1], view[1][1], view[2][1]};
        veekay::vec3 front{view[0][2], view[1][2], view[2][2]};

        right = veekay::vec3::normalized(right);
        up = veekay::vec3::normalized(up);
        front = veekay::vec3::normalized(front);

        float move_delta = movement_speed * delta_time;

        bool allow_keyboard_movement = !io.WantCaptureKeyboard;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::w))
                camera.position += front * move_delta;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::s))
                camera.position -= front * move_delta;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right * move_delta;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right * move_delta;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::q))
                camera.position += up * move_delta;

        if (allow_keyboard_movement && keyboard::isKeyDown(keyboard::Key::z))
                camera.position -= up * move_delta;

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
        };

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

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
