# Built inside the Audio2Face-3D SDK tree (see ../Dockerfile), next to the SDK's own samples,
# so it gets the SDK's compiler flags and links the single shared library it produces.
add_executable(a2f-worker)
set_target_properties(a2f-worker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
)
target_sources(a2f-worker PRIVATE main.cpp)
target_link_libraries(a2f-worker PRIVATE audio2x)
