#include "sgpt.h"

#include "acutest.h"

void test_init(void) {
  uint8_t mem_buffer[8];
  const sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 8,
      .mem_buffer = (void*)mem_buffer,
  });
  TEST_CHECK(ctx->mem_size == 8);
  TEST_CHECK(ctx->mem_buffer == mem_buffer);
  TEST_CHECK(ctx->n_objects == 0);
  TEST_CHECK(ctx->objects_begin == NULL);
  TEST_CHECK(ctx->objects_end == NULL);
}

void test_new_tensor(void) {
  uint8_t mem_buffer[4096];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 4096,
      .mem_buffer = (void*)mem_buffer,
  });

  const sgpt_tensor* a1 = sgpt_new_tensor_1d(ctx, SGPT_TYPE_F32, 4);
  TEST_CHECK(a1->n_dims == 1);
  TEST_CHECK(a1->ne[0] == 4);
  TEST_CHECK(a1->ne[1] == 1);
  TEST_CHECK(a1->ne[2] == 1);
  TEST_CHECK(a1->ne[3] == 1);
  TEST_CHECK(a1->nb[0] == 4);
  TEST_CHECK(a1->nb[1] == 16);
  TEST_CHECK(a1->nb[2] == 16);
  TEST_CHECK(a1->nb[3] == 16);

  const sgpt_tensor* a2 = sgpt_new_tensor_2d(ctx, SGPT_TYPE_F32, 2, 3);
  TEST_CHECK(a2->n_dims == 2);
  TEST_CHECK(a2->ne[0] == 2);
  TEST_CHECK(a2->ne[1] == 3);
  TEST_CHECK(a2->ne[2] == 1);
  TEST_CHECK(a2->ne[3] == 1);
  TEST_CHECK(a2->nb[0] == 4);
  TEST_CHECK(a2->nb[1] == 8);
  TEST_CHECK(a2->nb[2] == 24);
  TEST_CHECK(a2->nb[3] == 24);

  const sgpt_tensor* a3 = sgpt_new_tensor_3d(ctx, SGPT_TYPE_F32, 2, 3, 5);
  TEST_CHECK(a3->n_dims == 3);
  TEST_CHECK(a3->ne[0] == 2);
  TEST_CHECK(a3->ne[1] == 3);
  TEST_CHECK(a3->ne[2] == 5);
  TEST_CHECK(a3->ne[3] == 1);
  TEST_CHECK(a3->nb[0] == 4);
  TEST_CHECK(a3->nb[1] == 8);
  TEST_CHECK(a3->nb[2] == 24);
  TEST_CHECK(a3->nb[3] == 120);

  const sgpt_tensor* a4 = sgpt_new_tensor_4d(ctx, SGPT_TYPE_F32, 2, 3, 5, 7);
  TEST_CHECK(a4->n_dims == 4);
  TEST_CHECK(a4->ne[0] == 2);
  TEST_CHECK(a4->ne[1] == 3);
  TEST_CHECK(a4->ne[2] == 5);
  TEST_CHECK(a4->ne[3] == 7);
  TEST_CHECK(a4->nb[0] == 4);
  TEST_CHECK(a4->nb[1] == 8);
  TEST_CHECK(a4->nb[2] == 24);
  TEST_CHECK(a4->nb[3] == 120);
}

void test_set_i32(void) {
  uint8_t mem_buffer[4096];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 4096,
      .mem_buffer = (void*)mem_buffer,
  });

  int32_t value = 0;

  sgpt_tensor* a1 = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_set_i32_1d(a1, 0, 2);
  sgpt_set_i32_1d(a1, 1, 3);
  TEST_CHECK(sgpt_get_i32_1d(a1, 0) == 2);
  TEST_CHECK(sgpt_get_i32_1d(a1, 1) == 3);

  sgpt_tensor* a2 = sgpt_new_tensor_2d(ctx, SGPT_TYPE_I32, 2, 3);
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      value++;
      sgpt_set_i32_2d(a2, i, j, value);
    }
  }
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      value++;
      TEST_CHECK(sgpt_get_i32_2d(a2, i, j) == value);
    }
  }

  sgpt_tensor* a3 = sgpt_new_tensor_3d(ctx, SGPT_TYPE_I32, 2, 3, 5);
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 5; k++) {
        value++;
        sgpt_set_i32_3d(a3, i, j, k, value);
      }
    }
  }
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 5; k++) {
        value++;
        TEST_CHECK(sgpt_get_i32_3d(a3, i, j, k) == value);
      }
    }
  }

  sgpt_tensor* a4 = sgpt_new_tensor_4d(ctx, SGPT_TYPE_I32, 2, 3, 5, 7);
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 5; k++) {
        for (int l = 0; l < 7; l++) {
          value++;
          sgpt_set_i32_4d(a4, i, j, k, l, value);
        }
      }
    }
  }
  value = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 5; k++) {
        for (int l = 0; l < 7; l++) {
          value++;
          TEST_CHECK(sgpt_get_i32_4d(a4, i, j, k, l) == value);
        }
      }
    }
  }
}

void test_dup_tensor(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });
  const sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_F32, 4);
  const sgpt_tensor* b = sgpt_dup_tensor(ctx, a);
  TEST_CHECK(b->n_dims == 1);
  TEST_CHECK(b->ne[0] == 4);
  TEST_CHECK(b->ne[1] == 1);
  TEST_CHECK(b->ne[2] == 1);
  TEST_CHECK(b->ne[3] == 1);
  TEST_CHECK(b->nb[0] == 4);
  TEST_CHECK(b->nb[1] == 16);
  TEST_CHECK(b->nb[2] == 16);
  TEST_CHECK(b->nb[3] == 16);
}

void test_view_tensor(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  const sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  int32_t values[2] = {2, 3};
  memcpy(a->data, values, sizeof(int32_t) * 2);

  const sgpt_tensor* b = sgpt_view_tensor(ctx, a);
  TEST_CHECK(b->n_dims == 1);
  TEST_CHECK(b->ne[0] == 2);
  TEST_CHECK(b->ne[1] == 1);
  TEST_CHECK(b->ne[2] == 1);
  TEST_CHECK(b->ne[3] == 1);
  TEST_CHECK(b->nb[0] == 4);
  TEST_CHECK(b->nb[1] == 8);
  TEST_CHECK(b->nb[2] == 8);
  TEST_CHECK(b->nb[3] == 8);
  TEST_CHECK(a->data == b->data);
  TEST_CHECK(sgpt_get_i32_1d(b, 0) == 2);
  TEST_CHECK(sgpt_get_i32_1d(b, 1) == 3);
}

void test_dup(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* b = sgpt_dup(ctx, a);

  sgpt_cgraph gf = sgpt_build_forward(b);
  sgpt_set_i32_1d(a, 0, 1);
  sgpt_set_i32_1d(a, 1, 2);
  sgpt_graph_compute(ctx, &gf);

  TEST_CHECK(sgpt_get_i32_1d(b, 0) == 1);
  TEST_CHECK(sgpt_get_i32_1d(b, 1) == 2);
}

void test_dup_inplace(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* b = sgpt_dup_inplace(ctx, a);

  sgpt_cgraph gf = sgpt_build_forward(b);
  sgpt_set_i32_1d(a, 0, 1);
  sgpt_set_i32_1d(a, 1, 2);
  sgpt_graph_compute(ctx, &gf);

  TEST_CHECK(sgpt_get_i32_1d(b, 0) == 1);
  TEST_CHECK(sgpt_get_i32_1d(b, 1) == 2);
}

void test_add(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* b = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* c = sgpt_add(ctx, a, b);

  sgpt_cgraph gf = sgpt_build_forward(c);
  sgpt_set_i32_1d(a, 0, 1);
  sgpt_set_i32_1d(a, 1, 2);
  sgpt_set_i32_1d(b, 0, 3);
  sgpt_set_i32_1d(b, 1, 4);
  sgpt_graph_compute(ctx, &gf);

  TEST_CHECK(sgpt_get_i32_1d(c, 0) == 1 + 3);
  TEST_CHECK(sgpt_get_i32_1d(c, 1) == 2 + 4);
}

void test_add_inplace(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* b = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 2);
  sgpt_tensor* c = sgpt_add_inplace(ctx, a, b);

  sgpt_cgraph gf = sgpt_build_forward(c);
  sgpt_set_i32_1d(a, 0, 1);
  sgpt_set_i32_1d(a, 1, 2);
  sgpt_set_i32_1d(b, 0, 3);
  sgpt_set_i32_1d(b, 1, 4);
  sgpt_graph_compute(ctx, &gf);

  TEST_CHECK(sgpt_get_i32_1d(c, 0) == 1 + 3);
  TEST_CHECK(sgpt_get_i32_1d(c, 1) == 2 + 4);
}

TEST_LIST = {
    {"init", test_init},
    {"new_tensor", test_new_tensor},
    {"set_i32", test_set_i32},
    {"dup_tensor", test_dup_tensor},
    {"view_tensor", test_view_tensor},
    {"dup", test_dup},
    {"dup_inplace", test_dup_inplace},
    {"add", test_add},
    {"add_inplace", test_add_inplace},
    {NULL, NULL},
};
