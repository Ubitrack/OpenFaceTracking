#.rst:
# FindOpenFace
# ----------
#
# Try to find OpenFace (Tracking)
#
# Once done this will define::
#
#   OpenFace_FOUND          - True if OpenFace was found
#   OpenFace_INCLUDE_DIRS   - include directories for OpenFace
#   OpenFace_LIBRARIES      - link against this library to use OpenFace
#

#=============================================================================
# Copyright 2018 TUM FAR
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

find_path(OpenFace_ROOT_DIR
  NAMES OpenFace-license.txt
  PATHS
	"${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  PATH_SUFFIXES
    "Ubitrack_libs/VS2015-64/OpenFace"
	"libs/VS2015-64/OpenFace")
#message(STATUS "OpenFace_ROOT_DIR: ${OpenFace_ROOT_DIR}")

find_path(OpenFace_INCLUDE_BASE
  NAMES LandmarkDetector
  PATHS "${OpenFace_ROOT_DIR}/lib/local"
)
#message(STATUS "OpenFace_INCLUDE_BASE: ${OpenFace_INCLUDE_BASE}")

set(OpenFace_INCLUDE_DIRS 
	${OpenFace_INCLUDE_BASE}/FaceAnalyzer/include
	${OpenFace_INCLUDE_BASE}/GazeAnalyzer/include
	${OpenFace_INCLUDE_BASE}/LandmarkDetector/include
	${OpenFace_INCLUDE_BASE}/Utilities/include
)
#message(STATUS "OpenFace_INCLUDE_DIRS: ${OpenFace_INCLUDE_DIRS}")
	
if(WIN32)
	find_library(OpenFace_FaceAnalyser_LIB
	  NAMES FaceAnalyser
	  PATHS ${OpenFace_ROOT_DIR}/Release
	)
	find_library(OpenFace_GazeAnalyser_LIB
	  NAMES GazeAnalyser
	  PATHS ${OpenFace_ROOT_DIR}/Release
	)
	find_library(OpenFace_LandmarkDetector_LIB
	  NAMES LandmarkDetector
	  PATHS ${OpenFace_ROOT_DIR}/Release
	)
	find_library(OpenFace_Utilities_LIB
	  NAMES Utilities
	  PATHS ${OpenFace_ROOT_DIR}/Release
	)
else()
	find_library(OpenFace_FaceAnalyser_LIB
	  NAMES FaceAnalyser
	)
	find_library(OpenFace_GazeAnalyser_LIB
	  NAMES GazeAnalyser
	)
	find_library(OpenFace_LandmarkDetector_LIB
	  NAMES LandmarkDetector
	)
	find_library(OpenFace_Utilities_LIB
	  NAMES Utilities
	)
endif()
set(OpenFace_LIBRARIES "${OpenFace_FaceAnalyser_LIB} ${OpenFace_GazeAnalyser_LIB} ${OpenFace_LandmarkDetector_LIB} ${OpenFace_Utilities_LIB}")
#message(STATUS "OpenFace_LIBRARIES: ${OpenFace_LIBRARIES}")

set(OpenFace_LIBRARIES ${OpenFace_LIBRARY})

find_package_handle_standard_args(
  OpenFace
  FOUND_VAR OpenFace_FOUND
  REQUIRED_VARS 
    OpenFace_ROOT_DIR 
	OpenFace_INCLUDE_BASE 
	OpenFace_FaceAnalyser_LIB 
	OpenFace_GazeAnalyser_LIB 
	OpenFace_LandmarkDetector_LIB 
	OpenFace_Utilities_LIB 
	OpenFace_INCLUDE_DIRS
)
 