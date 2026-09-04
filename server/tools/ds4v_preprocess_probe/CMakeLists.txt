cmake_minimum_required(VERSION 3.20)
project(ds4v_preprocess_probe LANGUAGES CXX)

option(DS4V_PREPROCESS_WITH_CODECS "Build pinned JPEG/PNG decoder gate" ON)

add_executable(ds4v_preprocess_probe
    ds4v_preprocess_probe.cpp
    ../../src/deepseek4/deepseek4_vision_preprocess.cpp)
target_include_directories(ds4v_preprocess_probe PRIVATE ../../src/deepseek4)
target_compile_features(ds4v_preprocess_probe PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ds4v_preprocess_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(DS4V_PREPROCESS_WITH_CODECS)
    include(ExternalProject)
    include(FetchContent)

    set(DS4V_JPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-prefix)
    file(MAKE_DIRECTORY ${DS4V_JPEG_PREFIX}/include)
    ExternalProject_Add(libjpeg_turbo_external
        URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
        URL_HASH SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${DS4V_JPEG_PREFIX}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DENABLE_SHARED=OFF
            -DENABLE_STATIC=ON
            -DWITH_TOOLS=OFF
            -DWITH_TESTS=OFF
            -DWITH_SIMD=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 2
        BUILD_BYPRODUCTS ${DS4V_JPEG_PREFIX}/lib/libjpeg.a)
    add_library(ds4v_libjpeg STATIC IMPORTED GLOBAL)
    set_target_properties(ds4v_libjpeg PROPERTIES
        IMPORTED_LOCATION ${DS4V_JPEG_PREFIX}/lib/libjpeg.a
        INTERFACE_INCLUDE_DIRECTORIES ${DS4V_JPEG_PREFIX}/include)
    add_dependencies(ds4v_libjpeg libjpeg_turbo_external)

    FetchContent_Declare(lodepng
        URL https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
        URL_HASH SHA256=c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(lodepng)
    add_library(ds4v_lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)
    target_include_directories(ds4v_lodepng PUBLIC ${lodepng_SOURCE_DIR})

    target_sources(ds4v_preprocess_probe PRIVATE
        ../../src/deepseek4/deepseek4_vision_decode.cpp)
    target_compile_definitions(ds4v_preprocess_probe PRIVATE DS4V_PREPROCESS_WITH_CODECS=1)
    target_link_libraries(ds4v_preprocess_probe PRIVATE ds4v_libjpeg ds4v_lodepng)
endif()

enable_testing()
add_test(NAME ds4v_preprocess_self_test COMMAND ds4v_preprocess_probe --self-test)
