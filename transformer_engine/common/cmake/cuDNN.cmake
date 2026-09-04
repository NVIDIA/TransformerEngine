# Check if CUDNN:: targets already exist (integrated build)
if(TARGET CUDNN::cudnn AND TARGET CUDNN::cudnn_all)
    message(STATUS "cuDNN: Using existing CMake targets")
    set(CUDNN_FOUND ON CACHE INTERNAL "cuDNN Library Found")
    return()
endif()

message(STATUS "cuDNN: Searching for pre-built libraries")

add_library(CUDNN::cudnn_all INTERFACE IMPORTED)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_INCLUDE_PATH} ${CUDNN_INCLUDE_PATH} $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
)

if(CUDNN_INCLUDE_DIR)
    if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn_version.h)
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_HEADER_CONTENTS)
    else()
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
    endif()
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
    if(NOT CUDNN_VERSION_MAJOR)
        set(CUDNN_VERSION "?")
    else()
        set(CUDNN_VERSION
            "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()
    set(CUDNN_MAJOR_VERSION ${CUDNN_VERSION_MAJOR})
endif()

function(find_cudnn_library NAME)
    if(NOT "${ARGV1}" STREQUAL "OPTIONAL")
        set(_cudnn_required "REQUIRED")
    else()
        set(_cudnn_required "")
    endif()

    if(CUDNN_STATIC)
        set(library_names "${NAME}_static" "lib${NAME}_static_v${CUDNN_MAJOR_VERSION}.a")
    else()
        set(library_names ${NAME} "lib${NAME}.so.${CUDNN_MAJOR_VERSION}")
    endif()
    find_library(
        ${NAME}_LIBRARY
        NAMES ${library_names}
        NAMES_PER_DIR
        HINTS $ENV{CUDNN_LIBRARY_PATH} ${CUDNN_LIBRARY_PATH} $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib/x64 lib
        ${_cudnn_required}
    )

    if(${NAME}_LIBRARY)
        add_library(CUDNN::${NAME} UNKNOWN IMPORTED)
        set_target_properties(
            CUDNN::${NAME} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
            IMPORTED_LOCATION ${${NAME}_LIBRARY}
        )
        message(STATUS "${NAME} found at ${${NAME}_LIBRARY}.")
    else()
        message(STATUS "${NAME} not found.")
    endif()
endfunction()

if(NOT CUDNN_STATIC)
    find_cudnn_library(cudnn)
    set(CUDNN_LIBRARY_VAR cudnn_LIBRARY)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LIBRARY REQUIRED_VARS
    CUDNN_INCLUDE_DIR ${CUDNN_LIBRARY_VAR}
)

if(CUDNN_INCLUDE_DIR AND (CUDNN_STATIC OR cudnn_LIBRARY))
    message(STATUS "cuDNN: ${cudnn_LIBRARY}")
    message(STATUS "cuDNN: ${CUDNN_INCLUDE_DIR}")
    set(CUDNN_FOUND ON CACHE INTERNAL "cuDNN Library Found")
else()
    set(CUDNN_FOUND OFF CACHE INTERNAL "cuDNN Library Not Found")
endif()

target_include_directories(
    CUDNN::cudnn_all
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>
)

if(CUDNN_STATIC)
    add_library(CUDNN::cudnn INTERFACE IMPORTED)
    target_link_libraries(
        CUDNN::cudnn
        INTERFACE
        CUDNN::cudnn_all
    )
    find_package(ZLIB REQUIRED)
else()
    target_link_libraries(
        CUDNN::cudnn_all
        INTERFACE
        CUDNN::cudnn
    )
endif()

if(CUDNN_MAJOR_VERSION EQUAL 8)
    find_cudnn_library(cudnn_adv_infer)
    find_cudnn_library(cudnn_adv_train)
    find_cudnn_library(cudnn_cnn_infer)
    find_cudnn_library(cudnn_cnn_train)
    find_cudnn_library(cudnn_ops_infer)
    find_cudnn_library(cudnn_ops_train)

    target_link_libraries(
        CUDNN::cudnn_all
        INTERFACE
        CUDNN::cudnn_adv_train
        CUDNN::cudnn_ops_train
        CUDNN::cudnn_cnn_train
        CUDNN::cudnn_adv_infer
        CUDNN::cudnn_cnn_infer
        CUDNN::cudnn_ops_infer
    )
elseif(CUDNN_MAJOR_VERSION EQUAL 9)
    find_cudnn_library(cudnn_graph)
    find_cudnn_library(cudnn_engines_runtime_compiled)
    find_cudnn_library(cudnn_ops OPTIONAL)
    find_cudnn_library(cudnn_cnn OPTIONAL)
    find_cudnn_library(cudnn_adv OPTIONAL)
    find_cudnn_library(cudnn_engines_precompiled OPTIONAL)
    find_cudnn_library(cudnn_heuristic OPTIONAL)
    find_cudnn_library(cudnn_ext OPTIONAL)

    target_link_libraries(
        CUDNN::cudnn_all
        INTERFACE
        $<$<BOOL:${CUDNN_STATIC}>:-Wl,--whole-archive>
        CUDNN::cudnn_graph
        CUDNN::cudnn_engines_runtime_compiled
        CUDNN::cudnn_ops
        CUDNN::cudnn_cnn
        CUDNN::cudnn_adv
        $<$<NOT:$<BOOL:${CUDNN_SKIP_PRECOMPILED_LINK}>>:CUDNN::cudnn_engines_precompiled>
        CUDNN::cudnn_heuristic
        $<$<BOOL:${CUDNN_STATIC}>:-Wl,--no-whole-archive>
        $<$<BOOL:${CUDNN_STATIC}>:CUDA::cublasLt_static $<IF:$<TARGET_EXISTS:CUDA::nvrtc_static>,CUDA::nvrtc_static,CUDA::nvrtc> ZLIB::ZLIB>
    )
endif()
