# CMakeList.txt: OpenCVFaceDetection 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
# 导入opencv头文件
include_directories("D:/CommonDev/opencv/opencv4.1.2/opencv/build/include")
# 导入opencv动态库路径
link_directories("D:/CommonDev/opencv/opencv4.1.2/opencv/build/x64/vc15/lib")
# 将源代码添加到此项目的可执行文件。
add_executable (OpenCVFaceDetection "OpenCVFaceDetection.cpp" "OpenCVFaceDetection.h")
# 将名字为“opencv_world412d”的动态库预编译到“NeOpencvStructure”可执行文件
target_link_libraries(OpenCVFaceDetection opencv_world412d)
# TODO: 如有需要，请添加测试并安装目标。
