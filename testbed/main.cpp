#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec3 color;
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

        Mesh plane_mesh;
        Mesh cube_mesh;
        Mesh sphere_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture* texture;
        VkSampler texture_sampler;
}

struct OrbitSettings {
        float radius = 1.5f;
        float height = 0.5f;
        float angular_speed = 1.0f; // radians per second
};

struct OrbitState {
        OrbitSettings settings{};
        float accumulated_time = 0.0f;
        int direction = 1;
        bool paused = false;
};

inline namespace {
        OrbitState orbit_state{};
        double last_frame_time = 0.0;
        bool first_frame = true;
        size_t plane_model_index = std::numeric_limits<size_t>::max();
        size_t cube_model_index = std::numeric_limits<size_t>::max();
        size_t sphere_model_index = std::numeric_limits<size_t>::max();
}

float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
        auto translation = veekay::mat4::translation(position);
        auto scaling = veekay::mat4::scaling(scale);

        auto rotation_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto rotation_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rotation_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));

        auto rotation_matrix = rotation_z * rotation_y * rotation_x;

        return translation * rotation_matrix * scaling;
}

veekay::mat4 Camera::view() const {
        auto rotation_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto rotation_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rotation_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));

        auto rotation_matrix = rotation_z * rotation_y * rotation_x;
        auto rotation_inverse = veekay::mat4::transpose(rotation_matrix);
        auto translation = veekay::mat4::translation(-position);

        return rotation_inverse * translation;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

        // NOTE: Shaders expect P * V * M composition (gl_Position = VP * position).
        //       Keep that order so clip-space coordinates remain correct.
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
                                .format = VK_FORMAT_R32G32B32_SFLOAT,
                                .offset = offsetof(Vertex, color),
                        },
                        {
                                .location = 3,
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

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
                std::vector<Vertex> vertices = {
                        {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.3f, 0.4f, 0.3f}, {0.0f, 0.0f}},
                        {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.4f, 0.5f, 0.4f}, {1.0f, 0.0f}},
                        {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.6f, 0.5f}, {1.0f, 1.0f}},
                        {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.4f, 0.5f, 0.4f}, {0.0f, 1.0f}},
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
                        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                        {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                        {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                        {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},

                        {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                        {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                        {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
                        {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},

                        {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                        {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                        {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
                        {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

                        {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                        {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                        {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                        {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

                        {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                        {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                        {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                        {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                        {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                        {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                        {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
                        {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
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
                const uint32_t slices = 64;
                const uint32_t stacks = 32;
                const float radius = 0.4f;

                std::vector<Vertex> vertices;
                vertices.reserve((slices + 1) * (stacks + 1));

                for (uint32_t stack = 0; stack <= stacks; ++stack) {
                        float v = float(stack) / float(stacks);
                        float phi = v * float(M_PI);
                        float sin_phi = sinf(phi);
                        float cos_phi = cosf(phi);

                        for (uint32_t slice = 0; slice <= slices; ++slice) {
                                float u = float(slice) / float(slices);
                                float theta = u * 2.0f * float(M_PI);
                                float sin_theta = sinf(theta);
                                float cos_theta = cosf(theta);

                                veekay::vec3 normal{
                                        sin_phi * cos_theta,
                                        cos_phi,
                                        sin_phi * sin_theta,
                                };

                                veekay::vec3 position = normal * radius;
                                veekay::vec3 color = (normal + veekay::vec3{1.0f, 1.0f, 1.0f}) * 0.5f;

                                vertices.push_back(Vertex{
                                        .position = position,
                                        .normal = normal,
                                        .color = color,
                                        .uv = {u, 1.0f - v},
                                });
                        }
                }

                std::vector<uint32_t> indices;
                indices.reserve(stacks * slices * 6);

                for (uint32_t stack = 0; stack < stacks; ++stack) {
                        for (uint32_t slice = 0; slice < slices; ++slice) {
                                uint32_t first = stack * (slices + 1) + slice;
                                uint32_t second = first + slices + 1;

                                indices.push_back(first);
                                indices.push_back(second);
                                indices.push_back(first + 1);

                                indices.push_back(second);
                                indices.push_back(second + 1);
                                indices.push_back(first + 1);
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

        orbit_state = {};
        last_frame_time = 0.0;
        first_frame = true;

        // NOTE: Add models to scene
        models.clear();

        plane_model_index = models.size();
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .albedo_color = veekay::vec3{0.9f, 0.9f, 0.9f}
        });

        cube_model_index = models.size();
        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {0.0f, 0.0f, 0.0f},
                },
                .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
        });

        sphere_model_index = models.size();
        models.emplace_back(Model{
                .mesh = sphere_mesh,
                .transform = Transform{
                        .position = {orbit_state.settings.radius, orbit_state.settings.height, 0.0f},
                        .scale = {0.5f, 0.5f, 0.5f},
                },
                .albedo_color = veekay::vec3{0.8f, 0.9f, 1.0f}
        });
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

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

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
        float delta_time = 0.0f;
        if (first_frame) {
                last_frame_time = time;
                first_frame = false;
        } else {
                delta_time = static_cast<float>(time - last_frame_time);
                last_frame_time = time;
        }

        ImGui::Begin("Controls:");
        ImGui::SliderFloat("Orbit radius", &orbit_state.settings.radius, 0.5f, 5.0f);
        ImGui::SliderFloat("Orbit height", &orbit_state.settings.height, -2.0f, 2.0f);
        ImGui::SliderFloat("Orbit speed", &orbit_state.settings.angular_speed, 0.1f, 5.0f, "%.2f rad/s");
        if (ImGui::Button(orbit_state.paused ? "Resume orbit" : "Pause orbit")) {
                orbit_state.paused = !orbit_state.paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reverse direction")) {
                orbit_state.direction *= -1;
        }
        ImGui::Text("Direction: %s", orbit_state.direction > 0 ? "Counter-clockwise" : "Clockwise");
        ImGui::End();

        if (!ImGui::GetIO().WantCaptureMouse) {
                using namespace veekay::input;

                if (mouse::isButtonDown(mouse::Button::left)) {
                        const float sensitivity = 0.1f;
                        veekay::vec2 move_delta = mouse::cursorDelta();

                        camera.rotation.x = std::clamp(camera.rotation.x + move_delta.y * sensitivity, -89.0f, 89.0f);
                        camera.rotation.y += move_delta.x * sensitivity;

                        if (camera.rotation.y > 180.0f)
                                camera.rotation.y -= 360.0f;
                        else if (camera.rotation.y < -180.0f)
                                camera.rotation.y += 360.0f;

                        auto normalize = [](const veekay::vec3& v) {
                                float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                                if (length <= 0.0001f)
                                        return veekay::vec3{0.0f, 0.0f, 0.0f};
                                return veekay::vec3{v.x / length, v.y / length, v.z / length};
                        };

                        float pitch = toRadians(camera.rotation.x);
                        float yaw = toRadians(camera.rotation.y);

                        veekay::vec3 front = normalize({
                                std::cos(pitch) * std::sin(yaw),
                                std::sin(pitch),
                                std::cos(pitch) * std::cos(yaw),
                        });

                        const float half_pi = 0.5f * float(M_PI);

                        veekay::vec3 right = normalize({
                                std::sin(yaw - half_pi),
                                0.0f,
                                std::cos(yaw - half_pi),
                        });

                        veekay::vec3 up = normalize({
                                right.y * front.z - right.z * front.y,
                                right.z * front.x - right.x * front.z,
                                right.x * front.y - right.y * front.x,
                        });

                        const float move_speed = 3.0f * delta_time;

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
        }

        if (!orbit_state.paused) {
                orbit_state.accumulated_time += delta_time * static_cast<float>(orbit_state.direction);
        }

        Model& cube_model = models[cube_model_index];
        Model& sphere_model = models[sphere_model_index];

        float angle = orbit_state.settings.angular_speed * orbit_state.accumulated_time;
        float radius = std::max(0.1f, orbit_state.settings.radius);

        sphere_model.transform.position = {
                cube_model.transform.position.x + std::cos(angle) * radius,
                cube_model.transform.position.y + orbit_state.settings.height,
                cube_model.transform.position.z + std::sin(angle) * radius,
        };

        float tangent_angle = angle + 0.5f * float(M_PI);
        sphere_model.transform.rotation = {
                0.0f,
                -tangent_angle * 180.0f / float(M_PI),
                0.0f,
        };

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

        const size_t model_uniforms_alignment =
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

                uint32_t offset = static_cast<uint32_t>(i * model_uniforms_alignment);
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
