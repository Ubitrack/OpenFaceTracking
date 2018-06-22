IF(WIN32)
	set(the_description "The UbiTrack OpenFace Tracking Component")
	ut_add_component(OpenFaceTracking DEPS utcore utdataflow utvision )
	
	SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
	
	FIND_PACKAGE(OpenFace REQUIRED)
	
	ut_component_include_directories("src" ${OpenFace_INCLUDE_DIRS} ${TINYXML_INCLUDE_DIR} ${LOG4CPP_INCLUDE_DIR} ${BOOSTBINDINGS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
	ut_glob_component_sources(HEADERS "src/*.h" SOURCES "src/*.cpp")
	ut_create_single_component(${OpenFace_LIBRARIES})
	ut_install_utql_patterns()
ENDIF(WIN32)