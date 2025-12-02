add_executable(lab_3 main.cpp)

set_target_properties(lab_3 PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

if(MSVC)
    target_compile_options(lab_3 PRIVATE /wd4201)
    target_compile_definitions(lab_3 PRIVATE -D_USE_MATH_DEFINES)
endif()

target_link_libraries(lab_3 PRIVATE veekay Vulkan::Headers)

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
    set(LAB3_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(LAB3_SHADER_SOURCES
            ${LAB3_SHADER_DIR}/lab_3.vert
            ${LAB3_SHADER_DIR}/lab_3.frag
    )

    set(LAB3_SHADER_BINARIES)
    foreach(SHADER ${LAB3_SHADER_SOURCES})
        get_filename_component(FILE_NAME ${SHADER} NAME)
        set(OUTPUT_SPV ${LAB3_SHADER_DIR}/${FILE_NAME}.spv)
        add_custom_command(
                OUTPUT ${OUTPUT_SPV}
                COMMAND glslc ${SHADER} -o ${OUTPUT_SPV}
                DEPENDS ${SHADER}
                COMMENT "Compiling ${FILE_NAME}"
        )
        list(APPEND LAB3_SHADER_BINARIES ${OUTPUT_SPV})
    endforeach()

    add_custom_target(lab_3_shaders DEPENDS ${LAB3_SHADER_BINARIES})
    add_dependencies(lab_3 lab_3_shaders)
endif()

