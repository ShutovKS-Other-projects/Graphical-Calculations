cmake_minimum_required(VERSION 3.18)
project(SobelCUDA LANGUAGES CXX CUDA)

# --- OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- �������� ����������� ����
add_executable(sobel_cuda sobel_cuda.cu)

# --- ������� OpenCV
target_link_libraries(sobel_cuda PRIVATE ${OpenCV_LIBS})

# --- ����� ��� �����������
set_target_properties(sobel_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 60 70 75 80 86
)
