add_executable(sgpt_test sgpt_test.c)
target_include_directories(sgpt_test
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/tests
)
target_link_libraries(sgpt_test PRIVATE sgpt)
add_test(NAME sgpt_test COMMAND $<TARGET_FILE:sgpt_test>)
