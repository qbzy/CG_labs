add_executable(lab_4 main.cpp)

set_target_properties(lab_4 PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

if(MSVC)
    target_compile_options(lab_4 PRIVATE /wd4201)
    target_compile_definitions(lab_4 PRIVATE -D_USE_MATH_DEFINES)
endif()

target_link_libraries(lab_4 PRIVATE veekay Vulkan::Headers)

target_sources(lab_4 PRIVATE
    ${lodepng_SOURCE_DIR}/lodepng.cpp
)

target_include_directories(lab_4 PRIVATE
    ${lodepng_SOURCE_DIR}
)

# Компиляция шейдеров
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.vert.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.vert.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.frag.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.frag.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/shadow.vert.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadow.vert -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/shadow.vert.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadow.vert
)

add_custom_target(lab_4_shaders DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.vert.spv
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.frag.spv
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/shadow.vert.spv
)

add_dependencies(lab_4 lab_4_shaders)
