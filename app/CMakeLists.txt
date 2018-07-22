# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

INCLUDE_DIRECTORIES(src/main/cpp/include)
#INCLUDE_DIRECTORIES(src/main/cpp/include/CL)
#INCLUDE_DIRECTORIES(src/main/cpp/include/kernel)

AUX_SOURCE_DIRECTORY(src/main/cpp SRC_LIST)
MESSAGE(STATUS "CMAKE_SOURCE_PATH: ${CMAKE_SOURCE_PATH}")
MESSAGE(STATUS "PROJECT_SOURCE_PATH: ${PROJECT_SOURCE_PATH}")
MESSAGE(STATUS "PROJECT_BINARY_PATH: ${PROJECT_BINARY_PATH}")
MESSAGE(STATUS "SRC_LIST: ${SRC_LIST}")
MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE(STATUS "VERSION: ${CMAKE_MAJOR_VERSION} ${CMAKE_MINOR_VERSION}")
MESSAGE(STATUS "CMAKE_INCLUDE_PATH: ${CMAKE_INCLUDE_PATH}")
MESSAGE(STATUS "CMAKE_LIBRARY_PATH: ${CMAKE_LIBRARY_PATH}")

LINK_DIRECTORIES(libs)
ADD_DEFINITIONS(-fPIC -O2 -fexceptions -std=c++11 -DANDROID_LOGGER -DUSE_STUB_OPENCL -D__ANDROID__)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY jniLibs/${ANDROID_ABI})

add_library( # Sets the name of the library.
             clpeak

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${SRC_LIST}
             )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       clpeak

                       # Links the target library to the log library
                       # included in the NDK.
                       libOpenCL.a
                       ${log-lib} )