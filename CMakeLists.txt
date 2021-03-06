CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(insnet_benchmark)

SET(CMAKE_CXX_COMPILER /usr/bin/c++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH -Wall -Wno-sign-compare")

if(CMAKE_BUILD_TYPE MATCHES debug)
    ADD_DEFINITIONS(-DN3LDG_DEBUG)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse3 -std=c++17 -O1" )
elseif(CMAKE_BUILD_TYPE MATCHES profile)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pg -DNDEBUG" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -Ofast -march=native" )
endif()

INCLUDE_DIRECTORIES(3rd/cxxopts/include src)
INCLUDE_DIRECTORIES(3rd/inih)
INCLUDE_DIRECTORIES(3rd/tinyutf8)

if(USE_CUDA)
    add_definitions(-DUSE_GPU)
endif()


ADD_SUBDIRECTORY(3rd/insnet)
ADD_EXECUTABLE(main src/main.cc)
ADD_EXECUTABLE(what_lang src/what_lang.cc)

TARGET_LINK_LIBRARIES(main insnet)
TARGET_LINK_LIBRARIES(what_lang insnet)
