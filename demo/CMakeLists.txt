FILE(GLOB SOURCE_1 "${PROJECT_ROOT_PATH}/demo/*.cpp")

ADD_EXECUTABLE(test ${SOURCE_1})
TARGET_LINK_LIBRARIES(test sentianalysis GolaxyTAP pthread)
