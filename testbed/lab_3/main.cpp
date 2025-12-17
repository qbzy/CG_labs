#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

// Максимальное количество моделей и текстур
constexpr uint32_t max_models = 1024;
constexpr uint32_t max_textures = 8;

// Индексы текстур в массиве дескрипторов
constexpr int plane_texture_index = 0;
constexpr int cube_texture_red_index = 1;
constexpr int cube_texture_green_index = 2;

// Структура вершины: позиция, нормаль, UV-координаты
struct Vertex {
	veekay::vec3 position; // Позиция в 3D пространстве
	veekay::vec3 normal;   // Нормаль для освещения
	veekay::vec2 uv;       // Текстурные координаты (0-1)
};

// Uniform-буфер сцены (общий для всех объектов)
struct SceneUniforms {
	veekay::mat4 view_projection; // Матрица view * projection
};

// Uniform-буфер модели (индивидуальный для каждого объекта)
struct ModelUniforms {
	veekay::mat4 model;        // Матрица трансформации модели
	veekay::vec3 albedo_color; // Базовый цвет
	float use_texture;         // Флаг использования текстуры
	int texture_index;         // Индекс текстуры в массиве
	float _pad[3];             // Выравнивание для std140
};

// Меш содержит геометрию объекта
struct Mesh {
	veekay::graphics::Buffer* vertex_buffer; // Буфер вершин
	veekay::graphics::Buffer* index_buffer;  // Буфер индексов
	uint32_t indices;                        // Количество индексов
};

// Трансформация объекта в 3D пространстве
struct Transform {
	veekay::vec3 position = {};              // Позиция в мире
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f}; // Масштаб по осям
	veekay::vec3 rotation = {};              // Углы Эйлера (градусы)

	// Вычисляет матрицу модели (T * R * S)
	veekay::mat4 matrix() const;
};

// Модель = меш + трансформация + материал
struct Model {
	Mesh mesh;                 // Геометрия
	Transform transform;       // Позиция, поворот, масштаб
	veekay::vec3 albedo_color; // Базовый цвет
	bool use_texture = false;  // Использовать текстуру?
	int texture_index = 0;     // Индекс текстуры
};

// Камера для просмотра сцены
struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {}; // Позиция камеры
	veekay::vec3 rotation = {}; // Поворот (pitch, yaw, roll)

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// View-матрица (обратная трансформация)
	veekay::mat4 view() const;

	// Композиция view и projection
	veekay::mat4 view_projection(float aspect_ratio) const;

	// Направление взгляда камеры
	veekay::vec3 front_direction() const;
};

// Объекты сцены
inline namespace {
	// Камера с начальной позицией и поворотом
	Camera camera{
		.position = {0.0f, 0.5f, -3.0f},
		.rotation = {0.0f, -90.0f, 0.0f},
	};

	// Список всех моделей в сцене
	std::vector<Model> models;
}

// Вектор "вверх" в мировых координатах (Y-axis)
constexpr veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};

// Vulkan объекты для рендеринга
inline namespace {
	// Шейдерные модули (SPIR-V)
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	// Дескрипторы для передачи данных в шейдеры
	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	// Графический pipeline
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	// Uniform-буферы
	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	// Геометрия примитивов
	Mesh plane_mesh;
	Mesh cube_mesh;
	Mesh pyramid_mesh;

	// Fallback текстура (шахматная доска)
	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	// Текстуры сцены
	veekay::graphics::Texture* plane_texture;
	veekay::graphics::Texture* cube_texture_red;
	veekay::graphics::Texture* cube_texture_green;
	VkSampler texture_sampler; // Сэмплер для всех текстур

	// Массивы для дескрипторов
	std::vector<veekay::graphics::Texture*> descriptor_textures;
	std::vector<VkSampler> descriptor_samplers;
}

// Конвертирует градусы в радианы
float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// Ограничивает значение в диапазоне
template<typename T>
T clampValue(T value, T min_value, T max_value) {
	if (value < min_value) return min_value;
	if (value > max_value) return max_value;
	return value;
}

// Вычисляет матрицу модели из компонентов трансформации
veekay::mat4 Transform::matrix() const {
	// Матрицы трансляции и масштабирования
	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);

	// Конвертируем углы Эйлера в радианы
	const float rx = toRadians(rotation.x);
	const float ry = toRadians(rotation.y);
	const float rz = toRadians(rotation.z);

	// Предвычисляем тригонометрические функции
	const float cos_x = std::cos(rx);
	const float sin_x = std::sin(rx);
	const float cos_y = std::cos(ry);
	const float sin_y = std::sin(ry);
	const float cos_z = std::cos(rz);
	const float sin_z = std::sin(rz);

	// Матрица поворота вокруг оси X (pitch)
	veekay::mat4 rot_x = {
		1.0f,  0.0f,   0.0f,  0.0f,
		0.0f, cos_x, -sin_x,  0.0f,
		0.0f, sin_x,  cos_x,  0.0f,
		0.0f,  0.0f,   0.0f,  1.0f
	};

	// Матрица поворота вокруг оси Y (yaw)
	veekay::mat4 rot_y = {
		cos_y, 0.0f, sin_y, 0.0f,
		0.0f, 1.0f,  0.0f, 0.0f,
		-sin_y, 0.0f, cos_y, 0.0f,
		0.0f, 0.0f,  0.0f, 1.0f
	};

	// Матрица поворота вокруг оси Z (roll)
	veekay::mat4 rot_z = {
		cos_z, -sin_z, 0.0f, 0.0f,
		sin_z,  cos_z, 0.0f, 0.0f,
		0.0f,   0.0f, 1.0f, 0.0f,
		0.0f,   0.0f, 0.0f, 1.0f
	};

	// Комбинируем повороты (Z * Y * X)
	auto r = rot_z * rot_y * rot_x;

	// Итоговая матрица: Translation * Rotation * Scale
	return t * r * s;
}

// Вычисляет направление взгляда камеры из углов Эйлера
veekay::vec3 Camera::front_direction() const {
	const float yaw = toRadians(rotation.y);
	const float pitch = toRadians(rotation.x);
	const float cos_pitch = std::cos(pitch);

	// Сферические координаты -> декартовы
	veekay::vec3 front{
		std::cos(yaw) * cos_pitch,
		std::sin(pitch),
		std::sin(yaw) * cos_pitch,
	};

	return veekay::vec3::normalized(front);
}

// Вычисляет view-матрицу (look-at)
veekay::mat4 Camera::view() const {
	// Вычисляем базисные вектора камеры
	const veekay::vec3 front = front_direction();
	const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
	const veekay::vec3 up = veekay::vec3::cross(right, front);

	// View-матрица - обратная трансформация камеры
	return {
		right.x, up.x, -front.x, 0.0f,
		right.y, up.y, -front.y, 0.0f,
		right.z, up.z, -front.z, 0.0f,
		-veekay::vec3::dot(right, position),
		-veekay::vec3::dot(up, position),
		veekay::vec3::dot(front, position),
		1.0f
	};
}

// Композиция view и projection матриц
veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	// Инвертируем Y для Vulkan (система координат)
	projection[1][1] *= -1.0f;
	return view() * projection;
}

// Загружает скомпилированный шейдер  из файла
VkShaderModule loadShaderModule(const char* path) {
	// Открываем файл в бинарном режиме, курсор в конце
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << "\n";
		return nullptr;
	}

	// Получаем размер файла
	const std::streampos file_size = file.tellg();
	if (file_size <= 0) {
		std::cerr << "Shader file is empty: " << path << "\n";
		return nullptr;
	}

	// Читаем весь файл в буфер
	const size_t size = static_cast<size_t>(file_size);
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	// Создаем шейдерный модуль из SPIR-V байткода
	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

// Загружает PNG изображение и создает Vulkan текстуру
veekay::graphics::Texture* loadTextureFromFile(VkCommandBuffer cmd, const char* path) {
	std::vector<unsigned char> pixels;
	unsigned width = 0;
	unsigned height = 0;

	// Декодируем PNG файл в массив пикселей RGBA
	const unsigned error = lodepng::decode(pixels, width, height, path);
	if (error != 0) {
		std::cerr << "Failed to load texture from '" << path << "': "
		          << lodepng_error_text(error) << "\n";
		return nullptr;
	}

	if (pixels.empty() || width == 0 || height == 0) {
		std::cerr << "Texture '" << path << "' is empty\n";
		return nullptr;
	}

	// Создаем Vulkan текстуру из пикселей (загружается на GPU)
	return new veekay::graphics::Texture(cmd, width, height,
	                                     VK_FORMAT_R8G8B8A8_UNORM,
	                                     pixels.data());
}

// Инициализация всех Vulkan ресурсов
void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // Создание графического pipeline
		// Загружаем шейдеры
		vertex_shader_module = loadShaderModule("C:/Users/mrbor/CLionProjects/CG_labs/cmake-build-debug-visual-studio/testbed/lab_3/shaders/lab_3.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("C:/Users/mrbor/CLionProjects/CG_labs/cmake-build-debug-visual-studio/testbed/lab_3/shaders/lab_3.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		// Описываем стадии шейдеров в pipeline
		VkPipelineShaderStageCreateInfo stage_infos[2];

		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main", // Точка входа в шейдере
		};

		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// Описываем формат вершинного буфера (stride - размер вершины)
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// Описываем атрибуты вершин (position, normal, uv)
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // layout(location = 0) в шейдере
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT, // vec3
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1, // layout(location = 1)
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT, // vec3
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2, // layout(location = 2)
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT, // vec2 для UV
				.offset = offsetof(Vertex, uv),
			},
		};

		// Конфигурация входных данных вершинного шейдера
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// Топология примитивов (каждые 3 вершины = треугольник)
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// Настройки растеризации
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,      // Заполнять треугольники
			.cullMode = VK_CULL_MODE_BACK_BIT,        // Отсекать задние грани
			.frontFace = VK_FRONT_FACE_CLOCKWISE,     // Передние грани по часовой
			.lineWidth = 1.0f,
		};

		// Мультисэмплинг отключен (1 сэмпл на пиксель)
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		// Viewport - область рендеринга на экране
		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		// Scissor - прямоугольник отсечения
		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// Настройки depth-теста (проверка глубины)
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,                      // Включить depth-тест
			.depthWriteEnable = true,                     // Записывать в depth-буфер
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// Настройки смешивания цветов (blending отключен)
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// Создание пула дескрипторов
		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Для SceneUniforms
					.descriptorCount = 1,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, // Для ModelUniforms
					.descriptorCount = 1,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Для текстур
					.descriptorCount = max_textures,
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

		// Создание layout дескрипторов (описание структуры bindings)
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0, // binding = 0 в шейдере (SceneUniforms)
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1, // binding = 1 в шейдере (ModelUniforms, динамический)
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2, // binding = 2 в шейдере (массив текстур)
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = max_textures,
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

		// Выделение descriptor set из пула
		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to allocate Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// Создание pipeline layout (описывает внешние данные для pipeline)
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		// Создание графического pipeline (объединяет все состояния)
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

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// Создание uniform-буферов
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Динамический буфер для всех моделей (с выравниванием)
	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Создание fallback текстуры (шахматная доска 2x2)
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

		// Черно-розовая шахматная доска (BGRA формат)
		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	// Создание сэмплера для текстур сцены
	{
		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,              // Линейная фильтрация при увеличении
			.minFilter = VK_FILTER_LINEAR,              // Линейная фильтрация при уменьшении
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, // Повтор текстуры по U
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, // Повтор текстуры по V
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		};
		if (vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create texture sampler\n";
			veekay::app.running = false;
			return;
		}
	}

	// Лямбда для загрузки текстуры с fallback на процедурную
	auto load_or_fallback = [&](const char* path) -> veekay::graphics::Texture* {
		if (auto* loaded = loadTextureFromFile(cmd, path)) {
			return loaded;
		}

		// Если загрузка не удалась, создаем цветную шахматную доску 4x4
		std::cerr << "Falling back to procedural checkerboard texture for '"
		          << path << "'\n";
		uint32_t fallback_pixels[] = {
			0xffff0000, 0xff00ff00, 0xff0000ff, 0xffffff00,
			0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff0000,
			0xff0000ff, 0xffffff00, 0xffff0000, 0xff00ff00,
			0xffffff00, 0xffff0000, 0xff00ff00, 0xff0000ff,
		};

		return new veekay::graphics::Texture(
			cmd, 4, 4, VK_FORMAT_B8G8R8A8_UNORM, fallback_pixels);
	};

	// Загрузка текстур из файлов
	plane_texture = load_or_fallback("C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_3/shaders/texture.png");
	cube_texture_red = load_or_fallback("C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_3/shaders/texture_cube_green.png");
	cube_texture_green = load_or_fallback("C:/Users/mrbor/CLionProjects/CG_labs/testbed/lab_3/shaders/texture_cube_red.png");

	// Заполняем массивы текстур и сэмплеров для дескрипторов
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

	// Дополняем массивы до max_textures fallback текстурой
	while (descriptor_textures.size() < max_textures) {
		descriptor_textures.push_back(missing_texture);
		descriptor_samplers.push_back(missing_texture_sampler);
	}

	// Обновление дескрипторов (привязка буферов и текстур)
	{
		// Информация о uniform-буферах
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

		// Информация о текстурах (image view + sampler)
		std::vector<VkDescriptorImageInfo> image_infos;
		image_infos.reserve(max_textures);
		for (size_t i = 0; i < max_textures; ++i) {
			image_infos.push_back({
				.sampler = descriptor_samplers[i],
				.imageView = descriptor_textures[i]->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			});
		}

		// Записываем дескрипторы (binding 0, 1, 2)
		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0, // SceneUniforms
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1, // ModelUniforms (динамический)
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2, // Массив текстур
				.dstArrayElement = 0,
				.descriptorCount = max_textures,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = image_infos.data(),
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// Инициализация меша плоскости (пол)
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		// Вершины: позиция, нормаль вверх, UV-координаты
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		// Два треугольника образуют квад
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

	{
		std::vector<Vertex> vertices = {
			{{0.0f, 1.0f, 0.0f}, {0.0f, 0.707f, 0.707f}, {0.5f, 1.0f}},
			{{-0.5f, 0.0f, 0.5f}, {0.0f, 0.707f, 0.707f}, {0.0f, 0.0f}},
			{{0.5f, 0.0f, 0.5f}, {0.0f, 0.707f, 0.707f}, {1.0f, 0.0f}},

			{{0.0f, 1.0f, 0.0f}, {0.707f, 0.707f, 0.0f}, {0.5f, 1.0f}},
			{{0.5f, 0.0f, 0.5f}, {0.707f, 0.707f, 0.0f}, {0.0f, 0.0f}},
			{{0.5f, 0.0f, -0.5f}, {0.707f, 0.707f, 0.0f}, {1.0f, 0.0f}},

			{{0.0f, 1.0f, 0.0f}, {0.0f, 0.707f, -0.707f}, {0.5f, 1.0f}},
			{{0.5f, 0.0f, -0.5f}, {0.0f, 0.707f, -0.707f}, {0.0f, 0.0f}},
			{{-0.5f, 0.0f, -0.5f}, {0.0f, 0.707f, -0.707f}, {1.0f, 0.0f}},

			{{0.0f, 1.0f, 0.0f}, {-0.707f, 0.707f, 0.0f}, {0.5f, 1.0f}},
			{{-0.5f, 0.0f, -0.5f}, {-0.707f, 0.707f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, 0.0f, 0.5f}, {-0.707f, 0.707f, 0.0f}, {1.0f, 0.0f}},

			{{-0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
			{{0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2,
			3, 4, 5,
			6, 7, 8,
			9, 10, 11,
			12, 13, 14, 14, 15, 12,
		};

		pyramid_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		pyramid_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		pyramid_mesh.indices = uint32_t(indices.size());
	}

	// Добавление моделей в сцену
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{0.5f, 0.5f, 0.5f},
		.use_texture = false, // Плоскость без текстуры
		.texture_index = plane_texture_index,
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.0f, 0.5f, -1.5f},
		},
		.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
		.use_texture = true,
		.texture_index = cube_texture_red_index,
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {1.5f, 0.5f, -0.5f},
		},
		.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
		.use_texture = true,
		.texture_index = cube_texture_green_index,
	});

	models.emplace_back(Model{
		.mesh = pyramid_mesh,
		.transform = Transform{
			.position = {0.0f, 0.0f, 1.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f},
		.use_texture = true,
		.texture_index = plane_texture_index,
	});
}

// Освобождение всех Vulkan ресурсов
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	vkDestroySampler(device, texture_sampler, nullptr);
	delete cube_texture_green;
	delete cube_texture_red;
	delete plane_texture;

	delete pyramid_mesh.index_buffer;
	delete pyramid_mesh.vertex_buffer;

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
}

// Обновление логики каждый кадр
void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::End();

	// Управление камерой только если курсор не над ImGui окном
	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		// Вращение камеры мышью (ЛКМ)
		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();

			constexpr float sensitivity = 0.1f;
			camera.rotation.y -= move_delta.x * sensitivity; // Yaw
			camera.rotation.x -= move_delta.y * sensitivity; // Pitch
			camera.rotation.x = clampValue(camera.rotation.x, -89.0f, 89.0f); // Ограничение pitch
		}

		// Вычисляем базисные вектора камеры для движения
		const veekay::vec3 front = camera.front_direction();
		const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
		const veekay::vec3 up = veekay::vec3::cross(right, front);
		constexpr float move_speed = 0.1f;

		// WASD - движение вперед/назад/влево/вправо
		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position -= front * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position += front * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * move_speed;

		// Q/Z - движение вверх/вниз относительно камеры
		if (keyboard::isKeyDown(keyboard::Key::q))
			camera.position += up * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up * move_speed;

		// Space/Shift - движение вверх/вниз в мировых координатах
		if (keyboard::isKeyDown(keyboard::Key::space))
			camera.position -= world_up * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::left_shift))
			camera.position += world_up * move_speed;
	}

	// Обновление uniform-буферов
	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
	};

	// Подготовка данных для всех моделей
	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
		uniforms.use_texture = model.use_texture ? 1.0f : 0.0f;
		uniforms.texture_index = model.texture_index;
	}

	// Копируем данные сцены в mapped memory
	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	// Копируем данные моделей с учетом выравнивания (для динамического буфера)
	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

// Рендеринг кадра
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	// Начало записи команд рендеринга
	{
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	// Начало render pass (очистка буферов цвета и глубины)
	{
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}}; // Темно-серый фон
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

	// Привязка pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	// Кэшируем текущие буферы для избежания лишних привязок
	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	// Отрисовка всех моделей
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		// Привязываем vertex buffer только если изменился
		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		// Привязываем index buffer только если изменился
		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		// Привязываем дескрипторы с динамическим offset для текущей модели
		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                       pipeline_layout, 0, 1, &descriptor_set,
		                       1, &offset);

		// Отрисовка индексированной геометрии
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
