# CMakeList.txt : CMake project for Opencl-ex1, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project(Opencl-project-group8)

#Adding Opencl libs and include files to the proj
find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})


find_package(Boost 1.56 REQUIRED)

#adding the Boost libs to the proj
if (WIN32)
  set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
  set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

  include_directories(${BOOST_INC})
  link_directories(${BOOST_LIB})

elseif(UNIX)
  message("OS: Linux")
  include_directories(${Boost_INCLUDE_DIR})
endif()


#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "Core/*.cpp" "Core/*.c")
file(GLOB OPENCL_SRC "OpenCL/*.cpp" "OpenCL/*.c")

if (WIN32)
  file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
endif()

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add source to this project's executable.
#add_executable (Opencl-ex1 "src/OpenCLExercise2_Mandelbrot.cpp" ${CORE_SRC} ${OPENCL_SRC} )
add_executable (Opencl-project-group8 "src/OpenCLProject_NLM.cpp" ${CORE_SRC} ${OPENCL_SRC} )
#add_executable (Opencl-project-group8  "src/OpenCLExercise3_Sobel.cpp" ${CORE_SRC} ${OPENCL_SRC} )
#add_executable (Opencl-ex1 "src/OpenCLExercise1_Basics.cpp" ${CORE_SRC} ${OPENCL_SRC} )
target_include_directories (Opencl-project-group8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src")
target_link_libraries (Opencl-project-group8 ${OpenCL_LIBRARY} dl boost_system) #imagehlp)

# TODO: Add tests and install targets if needed.
