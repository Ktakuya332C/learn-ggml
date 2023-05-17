#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef enum sgpt_op {
  SGPT_OP_NONE = 0,
  SGPT_OP_DUP,
  SGPT_OP_ADD,
} sgpt_op;

typedef enum sgpt_type {
  SGPT_TYPE_F32 = 0,
  SGPT_TYPE_I32,
  SGPT_TYPE_COUNT,
} sgpt_type;

#define SGPT_MAX_DIMS 4
typedef struct sgpt_tensor {
  sgpt_type type;
  int n_dims;
  int64_t ne[SGPT_MAX_DIMS]; // the number of elements in each dimension
  size_t nb[SGPT_MAX_DIMS]; // stride of each dimension
  sgpt_op op;
  struct sgpt_tensor* src0;
  struct sgpt_tensor* src1;
  void* data;
} sgpt_tensor;

#define SGPT_MAX_NODES 4096
typedef struct sgpt_cgraph {
  int n_nodes;
  int n_leafs;
  struct sgpt_tensor* nodes[SGPT_MAX_NODES];
  struct sgpt_tensor* leafs[SGPT_MAX_NODES];
} sgpt_cgraph;

typedef struct sgpt_object {
  size_t offset;
  size_t size;
  struct sgpt_object* next;
} sgpt_object;

typedef struct sgpt_context {
  size_t mem_size;
  void* mem_buffer;
  int n_objects;
  sgpt_object* objects_begin;
  sgpt_object* objects_end;
} sgpt_context;

typedef struct sgpt_init_params {
  size_t mem_size;
  void* mem_buffer;
} sgpt_init_params;
sgpt_context* sgpt_init(sgpt_init_params params);

sgpt_tensor* sgpt_new_tensor_1d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0
);
sgpt_tensor* sgpt_new_tensor_2d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1
);
sgpt_tensor* sgpt_new_tensor_3d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2
);
sgpt_tensor* sgpt_new_tensor_4d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3
);

sgpt_tensor* sgpt_dup_tensor(sgpt_context* ctx, const sgpt_tensor* src);
sgpt_tensor* sgpt_view_tensor(sgpt_context* ctx, const sgpt_tensor* src);

int32_t sgpt_get_i32_1d(const sgpt_tensor* tensor, int i0);
int32_t sgpt_get_i32_2d(const sgpt_tensor* tensor, int i0, int i1);
int32_t sgpt_get_i32_3d(const sgpt_tensor* tensor, int i0, int i1, int i2);
int32_t sgpt_get_i32_4d(const sgpt_tensor* tensor, int i0, int i1, int i2, int i3);

void sgpt_set_i32_1d(sgpt_tensor* tensor, int i0, int32_t value);
void sgpt_set_i32_2d(sgpt_tensor* tensor, int i0, int i1, int32_t value);
void sgpt_set_i32_3d(sgpt_tensor* tensor, int i0, int i1, int i2, int32_t value);
void sgpt_set_i32_4d(sgpt_tensor* tensor, int i0, int i1, int i2, int i3, int32_t value);

sgpt_tensor* sgpt_dup(sgpt_context* ctx, sgpt_tensor* a);
sgpt_tensor* sgpt_dup_inplace(sgpt_context* ctx, sgpt_tensor* a);
sgpt_tensor* sgpt_add(sgpt_context* ctx, sgpt_tensor* a, sgpt_tensor* b);
sgpt_tensor* sgpt_add_inplace(sgpt_context* ctx, sgpt_tensor* a, sgpt_tensor* b);

sgpt_cgraph sgpt_build_forward(sgpt_tensor* tensor);
void sgpt_graph_compute(sgpt_context* ctx, sgpt_cgraph* cgraph);
