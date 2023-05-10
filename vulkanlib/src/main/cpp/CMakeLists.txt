# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.

project("vulkanlib")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        vulkanlib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        VulkanContext.cpp
        vulkanlib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")

find_library( # Sets the name of the path variable.
        liblog

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)
find_library(libandroid android)
find_library(libjnigraphics jnigraphics)
find_library(libvulkan vulkan)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        vulkanlib

        # Links the target library to the log library
        # included in the NDK.
        ${libandroid}
        ${liblog}
        ${libjnigraphics}
        ${libvulkan})