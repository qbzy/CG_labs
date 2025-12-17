add_executable(lab_3 main.cpp)

set_target_properties(lab_3 PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

if(MSVC)
    target_compile_options(lab_3 PRIVATE /wd4201)
    target_compile_definitions(lab_3 PRIVATE -D_USE_MATH_DEFINES)
endif()

target_link_libraries(lab_3 PRIVATE veekay Vulkan::Headers)

target_sources(lab_3 PRIVATE
    ${lodepng_SOURCE_DIR}/lodepng.cpp
)

target_include_directories(lab_3 PRIVATE
    ${lodepng_SOURCE_DIR}
)

# Компиляция шейдеров
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.vert.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lab_3.vert -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.vert.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lab_3.vert
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.frag.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lab_3.frag -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.frag.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lab_3.frag
)

add_custom_target(lab_3_shaders DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.vert.spv
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/lab_3.frag.spv
)

add_dependencies(lab_3 lab_3_shaders)
