
#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <string>
#include <lodepng.h>

namespace {

    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 4;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct DirectionalLight {
        veekay::vec3 direction;
        float _pad0;
        veekay::vec3 color;
        float intensity;
    };

    struct PointLight {
        veekay::vec3 position;
        float type;
        veekay::vec3 direction;
        float cutOff;
        veekay::vec3 color;
        float intensity;
        float constant;
        float linear;
        float quadratic;
        float outerCutOff;
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        DirectionalLight directional_light;
        veekay::vec3 camera_position;
        float _pad0;
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float shininess;
        veekay::vec3 specular_color;
        float use_albedo_texture;
        float use_specular_texture;
        float use_emissive_texture;
        float emissive_strength;
    };

    struct PointLightsSSBO {
        PointLight lights[max_point_lights];
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

        veekay::mat4 matrix() const;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
        size_t material_index;
    };

    struct Material {
        veekay::graphics::Texture* albedo;
        veekay::graphics::Texture* specular;
        veekay::graphics::Texture* emissive;
        VkDescriptorSet descriptor_set;
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        float yaw = 90.0f;
        float pitch = 0.0f;

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        veekay::mat4 view() const;
        veekay::mat4 view_projection(float aspect_ratio) const;
        veekay::vec3 get_front_vector() const;
        veekay::vec3 get_right_vector() const;
        veekay::vec3 get_up_vector() const;
    };

    inline namespace {
        Camera camera{
                .position = {0.0f, 0.5f, 5.0f}
        };

        std::vector<Model> models;
        DirectionalLight directional_light {
                .direction = {-0.5f, -1.0f, -0.5f},
                .color = {1.0f, 1.0f, 1.0f},
                .intensity = 0.5f
        };

        PointLightsSSBO point_lights_data;
    }

    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSetLayout material_descriptor_set_layout;
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_ssbo;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture* missing_texture;
        veekay::graphics::Texture* lenna_texture;
        veekay::graphics::Texture* specular_texture;
        veekay::graphics::Texture* emissive_texture;
        VkSampler texture_sampler;
        std::vector<Material> materials;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    float vector_length(const veekay::vec3& v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    veekay::vec3 normalize_vector(const veekay::vec3& v) {
        float len = vector_length(v);
        if (len > 0.0f) {
            return {v.x / len, v.y / len, v.z / len};
        }
        return v;
    }

    veekay::vec3 cross_product(const veekay::vec3& a, const veekay::vec3& b) {
        return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
        };
    }

    float dot_product(const veekay::vec3& a, const veekay::vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    veekay::mat4 Transform::matrix() const {
        auto t = veekay::mat4::translation(position);

        float rad_x = toRadians(rotation.x);
        float rad_y = toRadians(rotation.y);
        float rad_z = toRadians(rotation.z);

        float cos_x = cos(rad_x), sin_x = sin(rad_x);
        float cos_y = cos(rad_y), sin_y = sin(rad_y);
        float cos_z = cos(rad_z), sin_z = sin(rad_z);

        veekay::mat4 rot_x = {
                1.0f,  0.0f,   0.0f,  0.0f,
                0.0f, cos_x, -sin_x,  0.0f,
                0.0f, sin_x,  cos_x,  0.0f,
                0.0f,  0.0f,   0.0f,  1.0f
        };

        veekay::mat4 rot_y = {
                cos_y, 0.0f, sin_y, 0.0f,
                0.0f, 1.0f,  0.0f, 0.0f,
                -sin_y, 0.0f, cos_y, 0.0f,
                0.0f, 0.0f,  0.0f, 1.0f
        };

        veekay::mat4 rot_z = {
                cos_z, -sin_z, 0.0f, 0.0f,
                sin_z,  cos_z, 0.0f, 0.0f,
                0.0f,   0.0f, 1.0f, 0.0f,
                0.0f,   0.0f, 0.0f, 1.0f
        };

        auto r = rot_z * rot_y * rot_x;
        auto s = veekay::mat4::scaling(scale);

        return t * r * s;
    }

    veekay::vec3 Camera::get_front_vector() const {
        veekay::vec3 front;
        front.x = cos(toRadians(yaw)) * cos(toRadians(pitch));
        front.y = sin(toRadians(pitch));
        front.z = sin(toRadians(yaw)) * cos(toRadians(pitch));
        return normalize_vector(front);
    }

    veekay::vec3 Camera::get_right_vector() const {
        auto result = cross_product(get_front_vector(), {0.0f, 1.0f, 0.0f});
        return normalize_vector(result);
    }

    veekay::vec3 Camera::get_up_vector() const {
        auto result = cross_product(get_right_vector(), get_front_vector());
        return normalize_vector(result);
    }

    veekay::mat4 Camera::view() const {
        veekay::vec3 eye = position;
        veekay::vec3 camera_front = get_front_vector();
        veekay::vec3 center = camera_front + eye;
        veekay::vec3 up_param = get_up_vector();

        veekay::vec3 f = normalize_vector(center - eye);
        veekay::vec3 r = normalize_vector(cross_product(f, up_param));

        if (vector_length(r) < 0.001f) {
            r = normalize_vector(cross_product({0.0f, 0.0f, -1.0f}, f));
        }

        veekay::vec3 u = cross_product(r, f);

        veekay::mat4 result{};
        result[0][0] = r.x;
        result[1][0] = r.y;
        result[2][0] = r.z;

        result[0][1] = u.x;
        result[1][1] = u.y;
        result[2][1] = u.z;

        result[0][2] = -f.x;
        result[1][2] = -f.y;
        result[2][2] = -f.z;

        result[3][0] = -dot_product(r, eye);
        result[3][1] = -dot_product(u, eye);
        result[3][2] = dot_product(f, eye);
        result[3][3] = 1.0f;

        return result;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        projection[1][1] *= -1;
        return view() * projection;
    }

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
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    VkShaderModule loadShaderWithFallback(const char* filename) {
        namespace fs = std::filesystem;
        std::vector<fs::path> candidates = {
                fs::path("testbed/lab_3/shaders") / filename,
                fs::current_path() / "shaders" / filename,
                fs::current_path() / filename,
        };

        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                return loadShaderModule(candidate.string().c_str());
            }
        }

        return nullptr;
    }

    veekay::graphics::Texture* createTextureFromFile(VkCommandBuffer cmd, const std::filesystem::path& path) {
        std::vector<unsigned char> pixels;
        unsigned width = 0, height = 0;
        unsigned result = lodepng::decode(pixels, width, height, path.string());
        if (result) {
            std::cerr << "Failed to load texture from " << path << ": " << lodepng_error_text(result) << '\n';
            return nullptr;
        }

        if (width == 0 || height == 0) {
            std::cerr << "Loaded texture has invalid dimensions: " << path << '\n';
            return nullptr;
        }

        return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
    }

    veekay::graphics::Texture* createRadialTexture(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                                   const veekay::vec3& center_color, const veekay::vec3& edge_color,
                                                   float exponent) {
        std::vector<uint32_t> pixels(width * height);

        const veekay::vec2 center = {float(width) / 2.0f, float(height) / 2.0f};
        const float max_distance = std::sqrt(center.x * center.x + center.y * center.y);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                float dx = float(x) - center.x;
                float dy = float(y) - center.y;
                float distance = std::sqrt(dx * dx + dy * dy);
                float t = std::pow(1.0f - std::min(distance / max_distance, 1.0f), exponent);
                veekay::vec3 color = center_color * t + edge_color * (1.0f - t);
                uint32_t r = uint32_t(std::clamp(color.x, 0.0f, 1.0f) * 255.0f);
                uint32_t g = uint32_t(std::clamp(color.y, 0.0f, 1.0f) * 255.0f);
                uint32_t b = uint32_t(std::clamp(color.z, 0.0f, 1.0f) * 255.0f);
                pixels[y * width + x] = (0xffu << 24) | (b << 16) | (g << 8) | r;
            }
        }

        return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;

        { // Build graphics pipeline
            vertex_shader_module = loadShaderWithFallback("lab_3.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderWithFallback("lab_3.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2];

            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
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
                    .cullMode = VK_CULL_MODE_NONE,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

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

            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1,
                    .pViewports = &viewport,
                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

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
                                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                .descriptorCount = 64,
                        },
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 16,
                        .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

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
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetLayoutBinding bindings[] = {
                        {
                                .binding = 0,
                                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 2,
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

                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &material_descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan material descriptor set layout\n";
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

            VkDescriptorSetLayout set_layouts[] = {descriptor_set_layout, material_descriptor_set_layout};

            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 2,
                    .pSetLayouts = set_layouts,
            };

            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
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

            if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
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

        point_lights_ssbo = new veekay::graphics::Buffer(
                sizeof(PointLightsSSBO),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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
                    .maxAnisotropy = 8.0f,
                    .compareEnable = VK_FALSE,
                    .minLod = 0.0f,
                    .maxLod = 12.0f,
            };

            if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
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

            lenna_texture = createTextureFromFile(cmd, "assets/lenna.png");
            specular_texture = createRadialTexture(cmd, 256, 256,
                                                   {1.0f, 1.0f, 1.0f},
                                                   {0.05f, 0.05f, 0.1f},
                                                   3.0f);
            emissive_texture = createRadialTexture(cmd, 256, 256,
                                                   {1.5f, 0.6f, 0.2f},
                                                   {0.0f, 0.0f, 0.0f},
                                                   1.5f);
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
                            .buffer = point_lights_ssbo->buffer,
                            .offset = 0,
                            .range = sizeof(PointLightsSSBO),
                    }
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
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[2],
                    },
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        materials.emplace_back(Material{
                .albedo = lenna_texture ? lenna_texture : missing_texture,
                .specular = specular_texture ? specular_texture : missing_texture,
                .emissive = emissive_texture ? emissive_texture : missing_texture,
        });

        materials.emplace_back(Material{
                .albedo = missing_texture,
                .specular = specular_texture ? specular_texture : missing_texture,
                .emissive = emissive_texture ? emissive_texture : missing_texture,
        });

        for (Material& material : materials) {
            VkDescriptorSetAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &material_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(device, &info, &material.descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate material descriptor set\n";
                veekay::app.running = false;
                return;
            }

            VkDescriptorImageInfo image_infos[] = {
                    {
                            .sampler = texture_sampler,
                            .imageView = material.albedo ? material.albedo->view : missing_texture->view,
                            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    },
                    {
                            .sampler = texture_sampler,
                            .imageView = material.specular ? material.specular->view : missing_texture->view,
                            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    },
                    {
                            .sampler = texture_sampler,
                            .imageView = material.emissive ? material.emissive->view : missing_texture->view,
                            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    },
            };

            VkWriteDescriptorSet material_writes[] = {
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = material.descriptor_set,
                            .dstBinding = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .pImageInfo = &image_infos[0],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = material.descriptor_set,
                            .dstBinding = 1,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .pImageInfo = &image_infos[1],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = material.descriptor_set,
                            .dstBinding = 2,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .pImageInfo = &image_infos[2],
                    },
            };

            vkUpdateDescriptorSets(device, sizeof(material_writes) / sizeof(material_writes[0]), material_writes, 0, nullptr);
        }

        // Plane mesh initialization
        {
            std::vector<Vertex> vertices = {
                    {{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                    0, 2, 1, 2, 0, 3
            };

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = uint32_t(indices.size());
        }

        // Cube mesh initialization
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

        // Add models to scene
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{.position = {0.0f, 0.0f, 0.0f}},
                .albedo_color = {0.8f, 0.8f, 0.8f},
                .specular_color = {0.1f, 0.1f, 0.1f},
                .shininess = 32.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {-2.0f, 0.5f, -1.5f}, },
                .albedo_color = {1.0f, 0.0f, 0.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {1.5f, 0.5f, -0.5f}, },
                .albedo_color = {0.0f, 1.0f, 0.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {0.0f, 0.5f, 1.0f}, },
                .albedo_color = {0.0f, 0.0f, 1.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });

        point_lights_data.lights[0] = {
                .position = {1.5f, 1.0f, 4.5f},
                .type = 1.0f,
                .direction = {-1.5f, -0.5f, -3.0f},
                .cutOff = cos(toRadians(10.0f)),
                .color = {1.0f, 0.5f, 0.0f},
                .intensity = 2.5f,
                .constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,
                .outerCutOff = cos(toRadians(17.5f))
        };

        point_lights_data.lights[1] = {
                .position = {-4.0f, 0.5f, -1.5f},
                .type = 1.0f,
                .direction = {2.0f, 0.0f, 0.0f},
                .cutOff = cos(toRadians(12.5f)),
                .color = {0.0f, 0.5f, 1.0f},
                .intensity = 2.0f,
                .constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,
                .outerCutOff = cos(toRadians(17.5f))
        };

        point_lights_data.lights[2] = {
                .position = {1.5f, 4.0f, -3.0f},
                .type = 0.0f,
                .direction = {-1.5f, -1.0f, -7.0f},
                .cutOff = cos(toRadians(15.0f)),
                .color = {1.0f, 0.0f, 1.0f},
                .intensity = 7.0f,
                .constant = 1.0f, .linear = 0.07f, .quadratic = 0.017f,
                .outerCutOff = cos(toRadians(20.0f))
        };

        point_lights_data.lights[3] = {
                .position = {0.0f, 4.0f, 0.0f},
                .type = 1.0f,
                .direction = {0.0f, -1.0f, 0.0f},
                .cutOff = cos(toRadians(30.0f)),
                .color = {1.0f, 1.0f, 1.0f},
                .intensity = 1.0f,
                .constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,
                .outerCutOff = cos(toRadians(35.0f))
        };
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        vkDestroySampler(device, texture_sampler, nullptr);
        delete missing_texture;
        delete lenna_texture;
        delete specular_texture;
        delete emissive_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_ssbo;

        vkDestroyDescriptorSetLayout(device, material_descriptor_set_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        static double last_time = time;
        double delta_time = time - last_time;
        last_time = time;

        ImGui::Begin("Controls:");
        ImGui::Text("Directional Light");
        ImGui::DragFloat3("Direction", &directional_light.direction.x, 0.01f);
        ImGui::ColorEdit3("Color##dir", &directional_light.color.x);
        ImGui::SliderFloat("Intensity##dir", &directional_light.intensity, 0.0f, 5.0f);
        ImGui::Separator();
        ImGui::Text("Point Lights");
        for(int i = 0; i < max_point_lights; ++i) {
            std::string label = "Point Light " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                const char* light_types[] = { "Point", "Spot" };
                int current_type = (int)point_lights_data.lights[i].type;
                if (ImGui::Combo("Type", &current_type, light_types, 2)) {
                    point_lights_data.lights[i].type = (float)current_type;
                }

                ImGui::DragFloat3("Position", &point_lights_data.lights[i].position.x, 0.1f);

                if (current_type == 1) {
                    ImGui::DragFloat3("Direction", &point_lights_data.lights[i].direction.x, 0.01f);
                }

                ImGui::ColorEdit3("Color", &point_lights_data.lights[i].color.x);
                ImGui::SliderFloat("Intensity", &point_lights_data.lights[i].intensity, 0.0f, 10.0f);
                ImGui::TreePop();
            }
        }
        ImGui::End();

        if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();
                float sensitivity = 0.1f;
                camera.yaw   -= move_delta.x * sensitivity;
                camera.pitch += move_delta.y * sensitivity;

                if(camera.pitch > 89.0f) camera.pitch = 89.0f;
                if(camera.pitch < -89.0f) camera.pitch = -89.0f;
            }

            veekay::vec3 move_front = camera.get_front_vector();
            veekay::vec3 move_right = camera.get_right_vector();
            veekay::vec3 up = veekay::vec3{0.0f, 1.0f, 0.0f};
            float speed = 2.5f * float(delta_time);

            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position -= move_front * speed;
            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position += move_front * speed;
            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += move_right * speed;
            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= move_right * speed;
            if (keyboard::isKeyDown(keyboard::Key::space))
                camera.position -= up * speed;
            if (keyboard::isKeyDown(keyboard::Key::left_shift))
                camera.position += up * speed;
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .directional_light = directional_light,
                .camera_position = camera.position
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            ModelUniforms& uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.shininess = model.shininess;
            uniforms.specular_color = model.specular_color;
            uniforms.use_albedo_texture = 1.0f;
            uniforms.use_specular_texture = 1.0f;
            uniforms.use_emissive_texture = 1.0f;
            uniforms.emissive_strength = 1.3f - 0.1f * float(i);
        }

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
        *(PointLightsSSBO*)point_lights_ssbo->mapped_region = point_lights_data;

        const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            const ModelUniforms& uniforms = model_uniforms[i];

            char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        {
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(cmd, &info);
        }

        {
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

        const size_t model_uniforms_alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

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

            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

            const Material& material = materials[model.material_index % materials.size()];

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    1, 1, &material.descriptor_set, 0, nullptr);

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