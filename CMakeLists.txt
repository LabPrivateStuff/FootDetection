cmake_minimum_required(VERSION 2.8)

project( FootDetection )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


###############
### COMPILE ###
###############

# COMPILE EXAMPLES
add_executable( detect main.cpp)
################
###   LINK   ###
################
# FIRST LINK EXAMPLES:
target_link_libraries( detect ${OpenCV_LIBS} )
