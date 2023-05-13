add_library(sgpt SHARED)
target_sources(sgpt PRIVATE sgpt.c)
target_include_directories(sgpt PUBLIC ${PROJECT_SOURCE_DIR}/include)

