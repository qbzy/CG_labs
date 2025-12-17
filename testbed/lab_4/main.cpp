#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
// Define function pointer types for dynamic rendering if not already defined
#ifndef VK_KHR_dynamic_rendering
typedef struct VkRenderingAttachmentInfoKHR {
    VkStructureType          sType;
    const void*              pNext;
    VkImageView              imageView;
    VkImageLayout            imageLayout;
    VkResolveModeFlagBits    resolveMode;
    VkImageView              resolveImageView;
    VkImageLayout            resolveImageLayout;
    VkAttachmentLoadOp       loadOp;
    VkAttachmentStoreOp      storeOp;
    VkClearValue             clearValue;
} VkRenderingAttachmentInfoKHR;

typedef struct VkRenderingInfoKHR {
    VkStructureType                        sType;
    const void*                            pNext;
    VkRenderingFlags                       flags;
    VkRect2D                               renderArea;
    uint32_t                               layerCount;
    uint32_t                               viewMask;
    uint32_t                               colorAttachmentCount;
    const VkRenderingAttachmentInfoKHR*    pColorAttachments;
    const VkRenderingAttachmentInfoKHR*    pDepthAttachment;
    const VkRenderingAttachmentInfoKHR*    pStencilAttachment;
} VkRenderingInfoKHR;

typedef void (VKAPI_PTR *PFN_vkCmdBeginRenderingKHR)(VkCommandBuffer commandBuffer, const VkRenderingInfoKHR* pRenderingInfo);
typedef void (VKAPI_PTR *PFN_vkCmdEndRenderingKHR)(VkCommandBuffer commandBuffer);
typedef void (VKAPI_PTR *PFN_vkCmdBeginRendering)(VkCommandBuffer commandBuffer, const VkRenderingInfoKHR* pRenderingInfo);
typedef void (VKAPI_PTR *PFN_vkCmdEndRendering)(VkCommandBuffer commandBuffer);

#define VK_STRUCTURE_TYPE_RENDERING_INFO_KHR ((VkStructureType)1000044000)
#define VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR ((VkStructureType)1000044001)
#endif
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t shadow_map_size = 4096;
constexpr uint32_t max_textures = 8;

constexpr int plane_texture_index = 0;
constexpr int cube_texture_red_index = 1;
constexpr int cube_texture_green_index = 2;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::mat4 shadow_projection;
	veekay::vec3 view_position; float _pad0;
	veekay::vec3 ambient_light_intensity; float _pad1;
	veekay::vec3 sun_light_direction; float _pad2;
	veekay::vec3 sun_light_color; float _pad3;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float use_texture;
	veekay::vec3 specular_color; float shininess;
	veekay::vec4 texture_info;
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
	veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;
	bool use_texture = false;
	int texture_index = 0;
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
		.position = {0.0f, -1.0f, -6.0f},
		.rotation = {0.0f, 0.0f, 0.0f}
	};

	std::vector<Model> models;
	
	float rotation_sensitivity = 0.7f;
	veekay::vec3 ambient_light_intensity = {0.2f, 0.2f, 0.2f};
	veekay::vec3 sun_light_color = {1.0f, 1.0f, 1.0f};
	float light_pitch = 45.0f; // угол наклона света в градусах
	float light_yaw = 45.0f;    // угол поворота света в градусах
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

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* plane_texture;
	veekay::graphics::Texture* cube_texture_red;
	veekay::graphics::Texture* cube_texture_green;
	VkSampler texture_sampler;

	std::vector<veekay::graphics::Texture*> descriptor_textures;
	std::vector<VkSampler> descriptor_samplers;

	// Shadow mapping objects
	struct {
		VkFormat depth_image_format;
		VkImage depth_image;
		VkDeviceMemory depth_image_memory;
		VkImageView depth_image_view;
		VkShaderModule vertex_shader;
		VkDescriptorSetLayout descriptor_set_layout;
		VkDescriptorSet descriptor_set;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;
		veekay::graphics::Buffer* uniform_buffer;
		VkSampler sampler;
		veekay::mat4 matrix;
	} shadow;

	// Dynamic rendering function pointers
	PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR = nullptr;
	PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR = nullptr;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	auto s = veekay::mat4::scaling(scale);
	
	// Вращение вокруг осей X, Y, Z (в порядке Z * Y * X)
	auto rx = veekay::mat4::rotation(veekay::vec3{1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
	auto ry = veekay::mat4::rotation(veekay::vec3{0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
	auto rz = veekay::mat4::rotation(veekay::vec3{0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
	auto r = rz * ry * rx;
	
	auto t = veekay::mat4::translation(position);
	
	// Порядок: translation * rotation * scale (T * R * S)
	return t * r * s;
}

veekay::mat4 Camera::view() const {
	float pitch_rad = toRadians(rotation.x);
	float yaw_rad = toRadians(rotation.y);

	veekay::vec3 forward;
	forward.x = cosf(pitch_rad) * cosf(yaw_rad);
	forward.y = sinf(pitch_rad);
	forward.z = cosf(pitch_rad) * sinf(yaw_rad);
	forward = veekay::vec3::normalized(forward);

	veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};
	veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, world_up));
	veekay::vec3 up = veekay::vec3::cross(right, forward);

	veekay::mat4 result = veekay::mat4::identity();

	result[0][0] = right.x;
	result[1][0] = right.y;
	result[2][0] = right.z;

	result[0][1] = up.x;
	result[1][1] = up.y;
	result[2][1] = up.z;

	result[0][2] = -forward.x;
	result[1][2] = -forward.y;
	result[2][2] = -forward.z;

	veekay::mat4 translation = veekay::mat4::translation(-position);

	return result * translation;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// Helper functions for shadow mapping
veekay::mat4 lookAt(const veekay::vec3& eye, const veekay::vec3& center, const veekay::vec3& up) {
	veekay::vec3 f = veekay::vec3::normalized(center - eye);
	veekay::vec3 s = veekay::vec3::normalized(veekay::vec3::cross(f, up));
	veekay::vec3 u = veekay::vec3::cross(s, f);

	veekay::mat4 result = veekay::mat4::identity();
	result[0][0] = s.x;
	result[1][0] = s.y;
	result[2][0] = s.z;
	result[0][1] = u.x;
	result[1][1] = u.y;
	result[2][1] = u.z;
	result[0][2] = -f.x;
	result[1][2] = -f.y;
	result[2][2] = -f.z;
	result[3][0] = -veekay::vec3::dot(s, eye);
	result[3][1] = -veekay::vec3::dot(u, eye);
	result[3][2] = veekay::vec3::dot(f, eye);

	return result;
}

veekay::mat4 ortho(float left, float right, float bottom, float top, float near, float far) {
	// Standard orthographic projection (works for both OpenGL and Vulkan after NDC transform)
	// X and Y: [-1, 1], Z: will be transformed to [0, 1] in shader
	// IMPORTANT: For orthographic projection, w component should always be 1.0
	veekay::mat4 result{};
	result[0][0] = 2.0f / (right - left);
	result[1][1] = 2.0f / (top - bottom);
	result[2][2] = -2.0f / (far - near);  // Standard ortho (Z will be in [-1, 1] range)
	result[3][0] = -(right + left) / (right - left);
	result[3][1] = -(top + bottom) / (top - bottom);
	result[3][2] = -(far + near) / (far - near);
	result[3][3] = 1.0f;  // CRITICAL: w component must be 1.0 for orthographic projection
	return result;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "[ERROR] Failed to open shader file: " << path << "\n";
		std::cerr.flush();
		return nullptr;
	}
	
	size_t size = file.tellg();
	if (size == 0 || size == static_cast<size_t>(-1)) {
		std::cerr << "[ERROR] Invalid shader file size: " << path << " (size: " << size << ")\n";
		std::cerr.flush();
		file.close();
		return nullptr;
	}
	
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

veekay::graphics::Texture* loadTextureFromFile(VkCommandBuffer cmd, const char* path) {
	std::vector<unsigned char> pixels;
	unsigned width = 0;
	unsigned height = 0;
	const unsigned error = lodepng::decode(pixels, width, height, path);
	if (error != 0) {
		std::cerr << "[ERROR] Failed to load texture '" << path << "': "
		          << lodepng_error_text(error) << "\n";
		return nullptr;
	}

	if (pixels.empty() || width == 0 || height == 0) {
		std::cerr << "[ERROR] Texture '" << path << "' is empty\n";
		return nullptr;
	}

	return new veekay::graphics::Texture(cmd, width, height,
	                                     VK_FORMAT_R8G8B8A8_UNORM,
	                                     pixels.data());
}

void initialize(VkCommandBuffer cmd) {
	std::cerr << "[INIT] Starting initialization...\n";
	std::cerr.flush();
	
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	if (device == VK_NULL_HANDLE) {
		std::cerr << "[ERROR] Device is NULL!\n";
		std::cerr.flush();
		veekay::app.running = false;
		return;
	}
	if (physical_device == VK_NULL_HANDLE) {
		std::cerr << "[ERROR] Physical device is NULL!\n";
		std::cerr.flush();
		veekay::app.running = false;
		return;
	}

	std::cerr << "[INIT] Loading dynamic rendering extension functions...\n";
	// Load dynamic rendering extension functions
	// According to instruction, we use vkCmdBeginRenderingKHR and vkCmdEndRenderingKHR
	vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
		vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
	vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
		vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));

	if (!vkCmdBeginRenderingKHR || !vkCmdEndRenderingKHR) {
		std::cerr << "[ERROR] Failed to load VK_KHR_dynamic_rendering functions\n";
		std::cerr << "[ERROR] vkCmdBeginRenderingKHR: " << (vkCmdBeginRenderingKHR ? "OK" : "NULL") << "\n";
		std::cerr << "[ERROR] vkCmdEndRenderingKHR: " << (vkCmdEndRenderingKHR ? "OK" : "NULL") << "\n";
		std::cerr << "[ERROR] Make sure VK_KHR_dynamic_rendering extension is enabled\n";
		std::cerr.flush();
		veekay::app.running = false;
		return;
	}
	std::cerr << "[INIT] Dynamic rendering functions loaded successfully\n";

	{ // NOTE: Build graphics pipeline
		std::cerr << "[INIT] Loading shaders...\n";
		vertex_shader_module = loadShaderModule("C:/Users/mrbor/CLionProjects/CG_labs/cmake-build-debug-visual-studio/testbed/lab_4/shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "[ERROR] Failed to load Vulkan vertex shader from file: ./shaders/shader.vert.spv\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Vertex shader loaded\n";

		fragment_shader_module = loadShaderModule("C:/Users/mrbor/CLionProjects/CG_labs/cmake-build-debug-visual-studio/testbed/lab_4/shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "[ERROR] Failed to load Vulkan fragment shader from file: ./shaders/shader.frag.spv\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Fragment shader loaded\n";

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
					.descriptorCount = max_textures + 1,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 2, // Main descriptor set + shadow descriptor set
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			VkResult result = vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create Vulkan descriptor pool, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Descriptor pool created\n";
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
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = max_textures,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
			{
				.binding = 4,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			VkResult result = vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create Vulkan descriptor set layout, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Descriptor set layout created\n";
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			VkResult result = vkAllocateDescriptorSets(device, &info, &descriptor_set);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create Vulkan descriptor set, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Descriptor set allocated\n";
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		std::cerr << "[INIT] Creating pipeline layout...\n";
		VkResult result = vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
		if (result != VK_SUCCESS) {
			std::cerr << "[ERROR] Failed to create Vulkan pipeline layout, VkResult: " << result << "\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Pipeline layout created\n";
		
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
		std::cerr << "[INIT] Creating graphics pipeline...\n";
		VkResult pipeline_result = vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline);
		if (pipeline_result != VK_SUCCESS) {
			std::cerr << "[ERROR] Failed to create Vulkan pipeline, VkResult: " << pipeline_result << "\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Graphics pipeline created\n";
	}

	std::cerr << "[INIT] Creating uniform buffers...\n";
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	if (!scene_uniforms_buffer || !scene_uniforms_buffer->mapped_region) {
		std::cerr << "[ERROR] Failed to create scene uniforms buffer\n";
		veekay::app.running = false;
		return;
	}
	std::cerr << "[INIT] Scene uniforms buffer created\n";

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	if (!model_uniforms_buffer || !model_uniforms_buffer->mapped_region) {
		std::cerr << "[ERROR] Failed to create model uniforms buffer\n";
		veekay::app.running = false;
		return;
	}
	std::cerr << "[INIT] Model uniforms buffer created\n";

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 1.0f,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
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

	// NOTE: Load object textures
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = 16.0f,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 1000.0f,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};

		if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		plane_texture = loadTextureFromFile(cmd, "C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_4/shaders/texture.png");
		cube_texture_red = loadTextureFromFile(cmd, "C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_4/shaders/texture_cube_red.png");
		cube_texture_green = loadTextureFromFile(cmd, "C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_4/shaders/texture_cube_green.png");

		if (!plane_texture || !cube_texture_red || !cube_texture_green) {
			veekay::app.running = false;
			return;
		}

		descriptor_textures = {
			plane_texture,
			cube_texture_red,
			cube_texture_green,
		};
		descriptor_samplers = {
			texture_sampler,
			texture_sampler,
			texture_sampler,
		};

		while (descriptor_textures.size() < max_textures) {
			descriptor_textures.push_back(missing_texture);
			descriptor_samplers.push_back(missing_texture_sampler);
		}
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

		std::vector<VkDescriptorImageInfo> image_infos;
		image_infos.reserve(max_textures);
		for (size_t i = 0; i < max_textures; ++i) {
			image_infos.push_back({
				.sampler = descriptor_samplers[i],
				.imageView = descriptor_textures[i]->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			});
		}

		// Update descriptor sets without shadow texture (will be updated later after shadow objects are created)
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
				.descriptorCount = max_textures,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = image_infos.data(),
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
		std::cerr << "[INIT] Main descriptor set updated (without shadow texture)\n";
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
	.use_texture = true,
	.texture_index = plane_texture_index
});

models.emplace_back(Model{
	.mesh = cube_mesh,
	.transform = Transform{
		.position = {-2.0f, -0.5f, -1.5f},
	},
	.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
	.use_texture = true,
	.texture_index = cube_texture_red_index
});

models.emplace_back(Model{
	.mesh = cube_mesh,
	.transform = Transform{
		.position = {1.5f, -0.5f, -0.5f},
	},
	.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
	.use_texture = true,
	.texture_index = cube_texture_green_index
});

models.emplace_back(Model{
	.mesh = cube_mesh,
	.transform = Transform{
		.position = {0.0f, -0.5f, 1.0f},
	},
	.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f},
	.use_texture = true,
	.texture_index = cube_texture_red_index
});

	// NOTE: Initialize shadow mapping
	{
		std::cerr << "[INIT] Initializing shadow mapping...\n";
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
		
		if (physical_device == VK_NULL_HANDLE) {
			std::cerr << "[ERROR] Physical device is NULL in shadow mapping initialization!\n";
			veekay::app.running = false;
			return;
		}
		
		// Find depth format
		std::cerr << "[INIT] Finding depth format for shadow map...\n";
		VkFormat candidates[] = {
			VK_FORMAT_D32_SFLOAT,
			VK_FORMAT_D32_SFLOAT_S8_UINT,
			VK_FORMAT_D24_UNORM_S8_UINT,
		};

		shadow.depth_image_format = VK_FORMAT_UNDEFINED;
		for (const auto& f : candidates) {
			VkFormatProperties properties;
			vkGetPhysicalDeviceFormatProperties(physical_device, f, &properties);
			if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
				// Check if format supports comparison sampling (required for sampler2DShadow)
				if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
					shadow.depth_image_format = f;
					std::cerr << "[INIT] Selected depth format: " << f << " (supports depth attachment and sampling)\n";
					break;
				} else {
					std::cerr << "[INIT] Format " << f << " supports depth attachment but not sampling\n";
				}
			}
		}

		if (shadow.depth_image_format == VK_FORMAT_UNDEFINED) {
			std::cerr << "[ERROR] Failed to find suitable depth format for shadow map\n";
			veekay::app.running = false;
			return;
		}

		// Create shadow map image
		{
			std::cerr << "[INIT] Creating shadow map image (" << shadow_map_size << "x" << shadow_map_size << ")...\n";
			VkImageCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = shadow.depth_image_format,
				.extent = {shadow_map_size, shadow_map_size, 1},
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};

			VkResult result = vkCreateImage(device, &info, nullptr, &shadow.depth_image);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow map image, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow map image created\n";
		}

		// Allocate memory for shadow map
		{
			VkMemoryRequirements requirements;
			vkGetImageMemoryRequirements(device, shadow.depth_image, &requirements);

			VkPhysicalDeviceMemoryProperties properties;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

			uint32_t index = UINT_MAX;
			for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
				const VkMemoryType& type = properties.memoryTypes[i];
				if ((requirements.memoryTypeBits & (1 << i)) &&
				    (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
					index = i;
					break;
				}
			}

			if (index == UINT_MAX) {
				std::cerr << "[ERROR] Failed to find memory type for shadow map\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow map memory type found: " << index << "\n";

			VkMemoryAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				.allocationSize = requirements.size,
				.memoryTypeIndex = index,
			};

			VkResult result = vkAllocateMemory(device, &info, nullptr, &shadow.depth_image_memory);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to allocate memory for shadow map, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow map memory allocated\n";

			result = vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to bind memory for shadow map, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow map memory bound\n";
		}

		// Create shadow map image view
		{
			VkImageViewCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = shadow.depth_image,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = shadow.depth_image_format,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkResult result = vkCreateImageView(device, &info, nullptr, &shadow.depth_image_view);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow map image view, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow map image view created\n";
		}

		// Initialize shadow map image layout to SHADER_READ_ONLY_OPTIMAL
		{
			std::cerr << "[INIT] Initializing shadow map image layout...\n";
			VkImageMemoryBarrier barrier{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.image = shadow.depth_image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			                     0, 0, nullptr, 0, nullptr, 1, &barrier);
			std::cerr << "[INIT] Shadow map image layout initialized\n";
		}

		// Create shadow uniform buffer
		std::cerr << "[INIT] Creating shadow uniform buffer...\n";
		shadow.uniform_buffer = new veekay::graphics::Buffer(
			sizeof(veekay::mat4),
			nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		if (!shadow.uniform_buffer || !shadow.uniform_buffer->mapped_region) {
			std::cerr << "[ERROR] Failed to create shadow uniform buffer\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Shadow uniform buffer created\n";

		// Create shadow sampler
		{
			VkSamplerCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.compareEnable = VK_TRUE,
				.compareOp = VK_COMPARE_OP_LESS,
				.minLod = 0.0f,
				.maxLod = VK_LOD_CLAMP_NONE,
				.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
			};

			VkResult result = vkCreateSampler(device, &info, nullptr, &shadow.sampler);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow sampler, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow sampler created\n";
		}

		// Load shadow vertex shader
		std::cerr << "[INIT] Loading shadow vertex shader...\n";
		shadow.vertex_shader = loadShaderModule("C:/Users/mrbor/CLionProjects/CG_labs/cmake-build-debug-visual-studio/testbed/lab_4/shaders/shadow.vert.spv");
		if (!shadow.vertex_shader) {
			std::cerr << "[ERROR] Failed to load shadow vertex shader from file: ./shaders/shadow.vert.spv\n";
			veekay::app.running = false;
			return;
		}
		std::cerr << "[INIT] Shadow vertex shader loaded\n";

		// Create shadow descriptor set layout
		{
			std::cerr << "[INIT] Creating shadow descriptor set layout...\n";
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			VkResult result = vkCreateDescriptorSetLayout(device, &info, nullptr, &shadow.descriptor_set_layout);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow descriptor set layout, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow descriptor set layout created\n";
		}

		// Allocate shadow descriptor set
		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &shadow.descriptor_set_layout,
			};

			VkResult result = vkAllocateDescriptorSets(device, &info, &shadow.descriptor_set);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to allocate shadow descriptor set, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow descriptor set allocated\n";
		}

		// Update shadow descriptor set
		{
			VkDescriptorBufferInfo buffer_infos[] = {
				{
					.buffer = shadow.uniform_buffer->buffer,
					.offset = 0,
					.range = sizeof(veekay::mat4),
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
					.dstSet = shadow.descriptor_set,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffer_infos[0],
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = shadow.descriptor_set,
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.pBufferInfo = &buffer_infos[1],
				},
			};

			vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]), write_infos, 0, nullptr);
			std::cerr << "[INIT] Shadow descriptor set updated\n";
		}

		// Create shadow pipeline layout
		{
			std::cerr << "[INIT] Creating shadow pipeline layout...\n";
			VkPipelineLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &shadow.descriptor_set_layout,
			};

			VkResult result = vkCreatePipelineLayout(device, &info, nullptr, &shadow.pipeline_layout);
			if (result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow pipeline layout, VkResult: " << result << "\n";
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow pipeline layout created\n";
		}

		// Create shadow pipeline
		{
			VkPipelineShaderStageCreateInfo stage_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = shadow.vertex_shader,
				.pName = "main",
			};

			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			VkVertexInputAttributeDescription attributes[] = {
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vertex, position),
				},
			};

			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_FRONT_BIT, // Отсекаем передние треугольники
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.depthBiasEnable = VK_TRUE,
				.lineWidth = 1.0f,
			};

			VkPipelineMultisampleStateCreateInfo sample_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};

			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS,
			};

			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			};

			VkViewport viewport{
				.x = 0.0f, .y = 0.0f,
				.width = float(shadow_map_size),
				.height = float(shadow_map_size),
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};

			VkRect2D scissor{
				.offset = {0, 0},
				.extent = {shadow_map_size, shadow_map_size},
			};

			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.pViewports = &viewport,
				.scissorCount = 1,
				.pScissors = &scissor,
			};

			VkDynamicState dyn_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
				VK_DYNAMIC_STATE_DEPTH_BIAS,
			};

			VkPipelineDynamicStateCreateInfo dyn_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
				.pDynamicStates = dyn_states,
			};

			VkPipelineRenderingCreateInfoKHR format_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
				.depthAttachmentFormat = shadow.depth_image_format,
			};

			VkGraphicsPipelineCreateInfo pipeline_info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.pNext = &format_info,
				.stageCount = 1,
				.pStages = &stage_info,
				.pVertexInputState = &input_state_info,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.pDynamicState = &dyn_state_info,
				.layout = shadow.pipeline_layout,
			};

			VkResult shadow_pipeline_result = vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &shadow.pipeline);
			if (shadow_pipeline_result != VK_SUCCESS) {
				std::cerr << "[ERROR] Failed to create shadow pipeline, VkResult: " << shadow_pipeline_result << "\n";
				std::cerr.flush();
				veekay::app.running = false;
				return;
			}
			std::cerr << "[INIT] Shadow pipeline created\n";
		}
		
		// Update main descriptor set with shadow texture now that shadow objects are created
		{
			VkDescriptorImageInfo shadow_image_info{
				.sampler = shadow.sampler,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			
			VkWriteDescriptorSet shadow_write_info{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			};
			
			vkUpdateDescriptorSets(device, 1, &shadow_write_info, 0, nullptr);
			std::cerr << "[INIT] Main descriptor set updated with shadow texture\n";
		}
	}
	
	std::cerr << "[INIT] Initialization completed successfully!\n";
	std::cerr.flush();
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, texture_sampler, nullptr);
	delete cube_texture_green;
	delete cube_texture_red;
	delete plane_texture;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

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

	// Cleanup shadow mapping resources
	vkDestroyPipeline(device, shadow.pipeline, nullptr);
	vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);
	vkDestroySampler(device, shadow.sampler, nullptr);
	vkDestroyImageView(device, shadow.depth_image_view, nullptr);
	vkDestroyImage(device, shadow.depth_image, nullptr);
	vkFreeMemory(device, shadow.depth_image_memory, nullptr);
	delete shadow.uniform_buffer;
	vkDestroyShaderModule(device, shadow.vertex_shader, nullptr);
}

void update(double time) {
	// Calculate light direction from angles (used in both UI and rendering)
	float light_pitch_rad = toRadians(light_pitch);
	float light_yaw_rad = toRadians(light_yaw);
	
	ImGui::Begin("Controls:");
	ImGui::SliderFloat3("Ambient Light", &ambient_light_intensity.x, 0.0f, 1.0f);
	ImGui::SliderFloat3("Sun Light Color", &sun_light_color.x, 0.0f, 1.0f);
	ImGui::SliderFloat("Light Pitch", &light_pitch, -90.0f, 90.0f);
	ImGui::SliderFloat("Light Yaw", &light_yaw, -180.0f, 180.0f);
	
	// Диагностика
	ImGui::Text("Diagnostics:");
	veekay::vec3 light_dir;
	light_dir.x = cosf(light_yaw_rad) * cosf(light_pitch_rad);
	light_dir.y = sinf(light_pitch_rad);
	light_dir.z = sinf(light_yaw_rad) * cosf(light_pitch_rad);
	light_dir = veekay::vec3::normalized(light_dir);
	ImGui::Text("Light Direction: %.2f, %.2f, %.2f", light_dir.x, light_dir.y, light_dir.z);
	
	ImGui::End();

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();
			camera.rotation.y += move_delta.x * rotation_sensitivity;
			camera.rotation.x -= move_delta.y * rotation_sensitivity;
			if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f;
			if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
		}

		float pitch_rad = toRadians(camera.rotation.x);
		float yaw_rad = toRadians(camera.rotation.y);
		veekay::vec3 front;
		front.x = cosf(pitch_rad) * cosf(yaw_rad);
		front.y = sinf(pitch_rad);
		front.z = cosf(pitch_rad) * sinf(yaw_rad);
		front = veekay::vec3::normalized(front);

		veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};
		veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
		veekay::vec3 up = veekay::vec3::cross(right, front);

		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::q))
			camera.position += up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::left_shift))
			camera.position += world_up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::space))
			camera.position -= world_up * 0.1f;
	}

float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

	// --- Этап 1. Рассчитываем направление и матрицы освещения для последующего построения карты теней
	veekay::vec3 light_direction_world;
	light_direction_world.x = cosf(light_yaw_rad) * cosf(light_pitch_rad);
	light_direction_world.y = sinf(light_pitch_rad);
	light_direction_world.z = sinf(light_yaw_rad) * cosf(light_pitch_rad);
	light_direction_world = veekay::vec3::normalized(light_direction_world);

	veekay::vec3 light_target{0.0f, 0.0f, 0.0f};
	veekay::vec3 light_pos = light_target - light_direction_world * 50.0f;
	veekay::vec3 light_up{0.0f, 1.0f, 0.0f};

	veekay::mat4 light_view = lookAt(light_pos, light_target, light_up);
	veekay::mat4 light_proj = ortho(-30.0f, 30.0f, -30.0f, 30.0f, 0.1f, 200.0f);
	shadow.matrix = light_proj * light_view;
	shadow.matrix[3][0] = 0.0f;
	shadow.matrix[3][1] = 0.0f;
	shadow.matrix[3][2] = 0.0f;
	shadow.matrix[3][3] = 1.0f;

	// --- Этап 2. Обновляем uniform-буфер световой матрицей, чтобы шейдеры могли проецировать вершины в пространство света
	*reinterpret_cast<veekay::mat4*>(shadow.uniform_buffer->mapped_region) = shadow.matrix;
	
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.shadow_projection = shadow.matrix,
		.view_position = camera.position,
		.ambient_light_intensity = ambient_light_intensity,
		.sun_light_direction = light_direction_world,
		.sun_light_color = sun_light_color,
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
		uniforms.use_texture = model.use_texture ? 1.0f : 0.0f;
		uniforms.specular_color = model.specular_color;
		uniforms.shininess = model.shininess;
		uniforms.texture_info = {float(model.texture_index), 0.0f, 0.0f, 0.0f};
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
	if (cmd == VK_NULL_HANDLE) {
		std::cerr << "[ERROR] Command buffer is NULL in render()\n";
		return;
	}
	if (framebuffer == VK_NULL_HANDLE) {
		std::cerr << "[ERROR] Framebuffer is NULL in render()\n";
		return;
	}

	VkResult result = vkResetCommandBuffer(cmd, 0);
	if (result != VK_SUCCESS) {
		std::cerr << "[ERROR] Failed to reset command buffer, VkResult: " << result << "\n";
		return;
	}

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		result = vkBeginCommandBuffer(cmd, &info);
		if (result != VK_SUCCESS) {
			std::cerr << "[ERROR] Failed to begin command buffer, VkResult: " << result << "\n";
			return;
		}
	}

	// --- Этап 3. Рисуем сцену "глазами" источника света в offscreen depth-текстуру
	{
		if (shadow.depth_image == VK_NULL_HANDLE || shadow.depth_image_view == VK_NULL_HANDLE) {
			std::cerr << "[ERROR] Shadow depth resources are not initialized\n";
			return;
		}
		if (!vkCmdBeginRenderingKHR || !vkCmdEndRenderingKHR) {
			std::cerr << "[ERROR] Dynamic rendering functions are not available\n";
			return;
		}
		if (shadow.pipeline == VK_NULL_HANDLE) {
			std::cerr << "[ERROR] Shadow pipeline is NULL\n";
			return;
		}

		// Перед началом рендера переводим тень-буфер в layout для записи глубины
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.image = shadow.depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &barrier);

		VkRenderingAttachmentInfoKHR depth_attachment{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
			.imageView = shadow.depth_image_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.depthStencil = {1.0f, 0}},
		};

		VkRenderingInfoKHR rendering_info{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, shadow_map_size, shadow_map_size},
			.layerCount = 1,
			.pDepthAttachment = &depth_attachment,
		};

		vkCmdBeginRenderingKHR(cmd, &rendering_info);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);

		VkViewport viewport{0.0f, 0.0f, float(shadow_map_size), float(shadow_map_size), 0.0f, 1.0f};
		vkCmdSetViewport(cmd, 0, 1, &viewport);
		VkRect2D scissor = {0, 0, shadow_map_size, shadow_map_size};
		vkCmdSetScissor(cmd, 0, 1, &scissor);
		vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;
		const size_t model_uniforms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (!mesh.vertex_buffer || !mesh.vertex_buffer->buffer ||
			    !mesh.index_buffer || !mesh.index_buffer->buffer) {
				continue;
			}

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniforms_alignment;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline_layout,
			                       0, 1, &shadow.descriptor_set, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderingKHR(cmd);

		// После завершения рендера карта глубины снова переходит в режим чтения из пиксельного шейдера
		barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &barrier);
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

	if (pipeline == VK_NULL_HANDLE) {
		std::cerr << "[ERROR] Main pipeline is NULL\n";
		return;
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
	
	result = vkEndCommandBuffer(cmd);
	if (result != VK_SUCCESS) {
		std::cerr << "[ERROR] Failed to end command buffer, VkResult: " << result << "\n";
		return;
	}
}

} // namespace

int main() {
	std::cerr << "[MAIN] Application starting...\n";
	std::cerr.flush();
	
	int result = veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
	
	std::cerr << "[MAIN] Application exiting with code: " << result << "\n";
	std::cerr.flush();
	
	// Pause to see error messages on Windows
	#ifdef _WIN32
	std::cerr << "Press Enter to exit...\n";
	std::cin.get();
	#endif
	
	return result;
}
