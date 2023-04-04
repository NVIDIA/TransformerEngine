add_library(CUDNN::cudnn_all INTERFACE IMPORTED)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
)

function(find_cudnn_library NAME)
    string(TOUPPER ${NAME} UPPERCASE_NAME)

    find_library(
        ${UPPERCASE_NAME}_LIBRARY ${NAME}
        HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib/x64 lib
    )
    
    if(${UPPERCASE_NAME}_LIBRARY)
        add_library(CUDNN::${NAME} UNKNOWN IMPORTED)
        set_target_properties(
            CUDNN::${NAME} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
            IMPORTED_LOCATION ${${UPPERCASE_NAME}_LIBRARY}
        )
        message(STATUS "${NAME} found at ${${UPPERCASE_NAME}_LIBRARY}.")
    else()
        message(STATUS "${NAME} not found.")
    endif()


endfunction()

find_cudnn_library(cudnn)
find_cudnn_library(cudnn_adv_infer)
find_cudnn_library(cudnn_adv_train)
find_cudnn_library(cudnn_cnn_infer)
find_cudnn_library(cudnn_cnn_train)
find_cudnn_library(cudnn_ops_infer)
find_cudnn_library(cudnn_ops_train)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LIBRARY REQUIRED_VARS
    CUDNN_INCLUDE_DIR CUDNN_LIBRARY
)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)

    message(STATUS "cuDNN: ${CUDNN_LIBRARY}")
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

target_link_libraries(
    CUDNN::cudnn_all
    INTERFACE
    CUDNN::cudnn_adv_train
    CUDNN::cudnn_ops_train
    CUDNN::cudnn_cnn_train
    CUDNN::cudnn_adv_infer
    CUDNN::cudnn_cnn_infer
    CUDNN::cudnn_ops_infer
    CUDNN::cudnn 
)

