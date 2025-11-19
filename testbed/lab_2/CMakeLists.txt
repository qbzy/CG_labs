add_executable(lab_2 main.cpp)

set_target_properties(lab_2 PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

if(MSVC)
    target_compile_options(lab_2 PRIVATE /wd4201)
    target_compile_definitions(lab_2 PRIVATE -D_USE_MATH_DEFINES)
endif()

target_link_libraries(lab_2 PRIVATE veekay Vulkan::Headers)

FetchContent_Declare(
        lodepng
        GIT_REPOSITORY https://github.com/lvandeve/lodepng
)

FetchContent_MakeAvailable(lodepng)


target_sources(${PROJECT_NAME} PRIVATE
        ${lodepng_SOURCE_DIR}/lodepng.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${lodepng_SOURCE_DIR}
)



find_program(GLSLC_FOUND glslc)
if(GLSLC_FOUND)
    set(LAB2_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(LAB2_SHADER_SOURCES
        ${LAB2_SHADER_DIR}/lab_2.vert
        ${LAB2_SHADER_DIR}/lab_2.frag
    )

    set(LAB2_SHADER_BINARIES)
    foreach(SHADER ${LAB2_SHADER_SOURCES})
        get_filename_component(FILE_NAME ${SHADER} NAME)
        set(OUTPUT_SPV ${LAB2_SHADER_DIR}/${FILE_NAME}.spv)
        add_custom_command(
            OUTPUT ${OUTPUT_SPV}
            COMMAND glslc ${SHADER} -o ${OUTPUT_SPV}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${FILE_NAME}"
        )
        list(APPEND LAB2_SHADER_BINARIES ${OUTPUT_SPV})
    endforeach()

    add_custom_target(lab_2_shaders DEPENDS ${LAB2_SHADER_BINARIES})
    add_dependencies(lab_2 lab_2_shaders)
endif()
