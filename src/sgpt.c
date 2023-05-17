#include "sgpt.h"
#include <assert.h>

static const size_t SGPT_TYPE_SIZE[SGPT_TYPE_COUNT] = {
  [SGPT_TYPE_F32] = sizeof(float),
  [SGPT_TYPE_I32] = sizeof(int32_t),
};

static sgpt_context ctx;
sgpt_context* sgpt_init(sgpt_init_params params) {
  ctx = (sgpt_context){
    .mem_size = params.mem_size,
    .mem_buffer = params.mem_buffer,
    .n_objects = 0,
    .objects_begin = NULL,
    .objects_end = NULL,
  };
  return &ctx;
}

static inline bool sgpt_are_same_shape(
  const sgpt_tensor* a,
  const sgpt_tensor* b
) {
  static_assert(SGPT_MAX_DIMS == 4, "SPGT_MAX_DIMS != 4");
  return (
    (a->ne[0] == b->ne[0]) &&
    (a->ne[1] == b->ne[1]) &&
    (a->ne[2] == b->ne[2]) &&
    (a->ne[3] == b->ne[3])
  );
}

static sgpt_tensor* sgpt_new_tensor_impl(
  sgpt_context* ctx,
  sgpt_type type,
  int n_dims,
  const int64_t* ne,
  void* data
) {
  sgpt_object* obj_cur = ctx->objects_end;
  const size_t cur_offset = obj_cur == NULL ? 0 : obj_cur->offset;
  const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  const size_t cur_end = cur_offset + cur_size;
  char* const mem_buffer = ctx->mem_buffer;
  
  size_t size_needed = 0;
  if (data == NULL) {
    size_needed += SGPT_TYPE_SIZE[type];
    for (int i=0; i<n_dims; i++) {
      size_needed *= ne[i];
    }
  }
  size_needed += sizeof(sgpt_tensor);
  assert(cur_end + sizeof(sgpt_object) + size_needed <= ctx->mem_size);
  
  sgpt_object* const obj_new = (sgpt_object*)(mem_buffer + cur_end);
  *obj_new = (sgpt_object){
    .offset = cur_end + sizeof(sgpt_object),
    .size = size_needed,
    .next = NULL,
  };
  if (obj_cur == NULL) {
    ctx->objects_begin = obj_new;
  } else {
    obj_cur->next = obj_new;
  }
  ctx->objects_end = obj_new;
  
  sgpt_tensor* const result = (sgpt_tensor*)(mem_buffer + obj_new->offset);
  *result = (sgpt_tensor){
    .type = type,
    .n_dims = n_dims,
    .ne = { 1, 1, 1, 1 }, // placeholder
    .nb = { 0, 0, 0, 0 }, // placeholder
    .op = SGPT_OP_NONE,
    .src0 = NULL,
    .src1 = NULL,
    .data = data == NULL ? (void*)(result+1) : data,
  };
  for (int i=0; i<n_dims; i++) result->ne[i] = ne[i];
  result->nb[0] = SGPT_TYPE_SIZE[type];
  result->nb[1] = result->nb[0] * result->ne[0];
  result->nb[2] = result->nb[1] * result->ne[1];
  result->nb[3] = result->nb[2] * result->ne[2];
  
  ctx->n_objects++;
  
  return result;
}

static sgpt_tensor* sgpt_new_tensor(
  sgpt_context* ctx,
  sgpt_type type,
  int n_dims,
  const int64_t* ne
) {
  return sgpt_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

sgpt_tensor* sgpt_new_tensor_1d(
  sgpt_context* ctx,
  sgpt_type type, int64_t ne0) {
  return sgpt_new_tensor(ctx, type, 1, &ne0);
}

sgpt_tensor* sgpt_new_tensor_2d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1
) {
  const int64_t ne[2] = { ne0, ne1 };
  return sgpt_new_tensor(ctx, type, 2, ne);
}

sgpt_tensor* sgpt_new_tensor_3d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2
) {
  const int64_t ne[3] = { ne0, ne1, ne2 };
  return sgpt_new_tensor(ctx, type, 3, ne);
}

sgpt_tensor* sgpt_new_tensor_4d(
  sgpt_context* ctx,
  sgpt_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3
) {
  const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
  return sgpt_new_tensor(ctx, type, 4, ne);
}

sgpt_tensor* sgpt_dup_tensor(sgpt_context* ctx, const sgpt_tensor* src) {
  return sgpt_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}

sgpt_tensor* sgpt_view_tensor(sgpt_context* ctx, const sgpt_tensor* src) {
  return sgpt_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);
}

int32_t sgpt_get_i32(const sgpt_tensor* tensor, int n_dims, const int* idxs) {
  assert(tensor->type == SGPT_TYPE_I32);
  int loc = 0;
  for (int i=0; i<n_dims; i++) {
    assert(idxs[i] <= tensor->ne[i]);
    loc += idxs[i] * tensor->nb[i];
  }
  return ((int32_t*)(tensor->data + loc))[0];
}

int32_t sgpt_get_i32_1d(const sgpt_tensor* tensor, int i0) {
  return sgpt_get_i32(tensor, 1, &i0);
}

int32_t sgpt_get_i32_2d(const sgpt_tensor* tensor, int i0, int i1) {
  const int idxs[2] = { i0, i1 };
  return sgpt_get_i32(tensor, 2, idxs);
}

int32_t sgpt_get_i32_3d(const sgpt_tensor* tensor, int i0, int i1, int i2) {
  const int idxs[3] = { i0, i1, i2 };
  return sgpt_get_i32(tensor, 3, idxs);
}

int32_t sgpt_get_i32_4d(const sgpt_tensor* tensor, int i0, int i1, int i2, int i3) {
  const int idxs[4] = { i0, i1, i2, i3 };
  return sgpt_get_i32(tensor, 4, idxs);
}

void sgpt_set_i32(const sgpt_tensor* tensor, int n_dims, const int* idxs, int32_t value) {
  assert(tensor->type == SGPT_TYPE_I32);
  int loc = 0;
  for (int i=0; i<n_dims; i++) {
    assert(idxs[i] <= tensor->ne[i]);
    loc += idxs[i] * tensor->nb[i];
  }
  ((int32_t*)(tensor->data + loc))[0] = value;
}

void sgpt_set_i32_1d(sgpt_tensor* tensor, int i0, int32_t value) {
  return sgpt_set_i32(tensor, 1, &i0, value);
}

void sgpt_set_i32_2d(sgpt_tensor* tensor, int i0, int i1, int32_t value) {
  const int idxs[2] = { i0, i1 };
  return sgpt_set_i32(tensor, 2, idxs, value);
}

void sgpt_set_i32_3d(sgpt_tensor* tensor, int i0, int i1, int i2, int32_t value) {
  const int idxs[3] = { i0, i1, i2 };
  return sgpt_set_i32(tensor, 3, idxs, value);
}

void sgpt_set_i32_4d(sgpt_tensor* tensor, int i0, int i1, int i2, int i3, int32_t value) {
  const int idxs[4] = { i0, i1, i2, i3 };
  return sgpt_set_i32(tensor, 4, idxs, value);
}

static sgpt_tensor* sgpt_dup_impl(sgpt_context* ctx, sgpt_tensor* a, bool inplace) {
  sgpt_tensor* result = inplace ? sgpt_view_tensor(ctx, a) : sgpt_dup_tensor(ctx, a);
  result->op = SGPT_OP_DUP;
  result->src0 = a;
  result->src1 = NULL;
  return result;
}

sgpt_tensor* sgpt_dup(sgpt_context* ctx, sgpt_tensor* a) {
  return sgpt_dup_impl(ctx, a, false);
}

sgpt_tensor* sgpt_dup_inplace(sgpt_context* ctx, sgpt_tensor* a) {
  return sgpt_dup_impl(ctx, a, true);
}

static sgpt_tensor* sgpt_add_impl(sgpt_context* ctx, sgpt_tensor* a, sgpt_tensor* b, bool inplace) {
  assert(sgpt_are_same_shape(a, b));
  sgpt_tensor* result = inplace ? sgpt_view_tensor(ctx, a) : sgpt_dup_tensor(ctx, a);
  result->op = SGPT_OP_ADD;
  result->src0 = a;
  result->src1 = b;
  return result;
}

sgpt_tensor* sgpt_add(sgpt_context* ctx, sgpt_tensor* a, sgpt_tensor* b) {
  return sgpt_add_impl(ctx, a, b, false);
}

sgpt_tensor* sgpt_add_inplace(sgpt_context* ctx, sgpt_tensor* a, sgpt_tensor* b) {
  return sgpt_add_impl(ctx, a, b, true);
}

static void sgpt_visit_parents(sgpt_cgraph* cgraph, sgpt_tensor* node) {
  for (int i=0; i<cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) return;
  }
  for (int i=0; i<cgraph->n_leafs; i++) {
    if (cgraph->leafs[i] == node) return;
  }
  if (node->src0) sgpt_visit_parents(cgraph, node->src0);
  if (node->src1) sgpt_visit_parents(cgraph, node->src1);
  if (node->op == SGPT_OP_NONE) {
    assert(cgraph->n_leafs < SGPT_MAX_NODES);
    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
  } else {
    assert(cgraph->n_nodes < SGPT_MAX_NODES);
    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->n_nodes++;
  }
}

sgpt_cgraph sgpt_build_forward(sgpt_tensor* tensor) {
  sgpt_cgraph result = {
    .n_nodes = 0,
    .n_leafs = 0,
    .nodes = { NULL },
    .leafs = { NULL },
  };
  sgpt_visit_parents(&result, tensor);
  if (result.n_nodes > 0) assert(result.nodes[result.n_nodes-1] == tensor);
  return result;
}

static void sgpt_compute_forward_dup_f32(sgpt_tensor* src0, sgpt_tensor* dst) {
  static_assert(SGPT_MAX_DIMS == 4, "SPGT_MAX_DIMS != 4");
  assert(src0->ne[0] == dst->ne[0]);
  assert(src0->ne[1] == dst->ne[1]);
  assert(src0->ne[2] == dst->ne[2]);
  assert(src0->ne[3] == dst->ne[3]);
  for (int i3=0; i3<src0->ne[3]; i3++) {
    size_t loc3 = src0->nb[3] * i3;
    for (int i2=0; i2<src0->ne[2]; i2++) {
      size_t loc2 = loc3 + src0->nb[2] * i2;
      for (int i1=0; i1<src0->ne[1]; i1++) {
        size_t loc1 = loc2 + src0->nb[1] * i1;
        for (int i0=0; i0<src0->ne[0]; i0++) {
          size_t loc0 = loc1 + src0->nb[0] * i0;
          ((float*)(dst->data + loc0))[0] = ((float*)(src0->data + loc0))[0];
        }
      }
    }
  }
}

static void sgpt_compute_forward_dup_i32(sgpt_tensor* src0, sgpt_tensor* dst) {
  static_assert(SGPT_MAX_DIMS == 4, "SPGT_MAX_DIMS != 4");
  assert(src0->ne[0] == dst->ne[0]);
  assert(src0->ne[1] == dst->ne[1]);
  assert(src0->ne[2] == dst->ne[2]);
  assert(src0->ne[3] == dst->ne[3]);
  for (int i3=0; i3<src0->ne[3]; i3++) {
    size_t loc3 = src0->nb[3] * i3;
    for (int i2=0; i2<src0->ne[2]; i2++) {
      size_t loc2 = loc3 + src0->nb[2] * i2;
      for (int i1=0; i1<src0->ne[1]; i1++) {
        size_t loc1 = loc2 + src0->nb[1] * i1;
        for (int i0=0; i0<src0->ne[0]; i0++) {
          size_t loc0 = loc1 + src0->nb[0] * i0;
          ((int32_t*)(dst->data + loc0))[0] = ((int32_t*)(src0->data + loc0))[0];
        }
      }
    }
  }
}

static void sgpt_compute_forward_dup(sgpt_tensor* src0, sgpt_tensor* dst) {
  assert(src0->type == dst->type);
  switch (src0->type) {
    case SGPT_TYPE_F32:
      sgpt_compute_forward_dup_f32(src0, dst);
      break;
    case SGPT_TYPE_I32:
      sgpt_compute_forward_dup_i32(src0, dst);
      break;
    default:
      assert(false);
  }
}

static void sgpt_compute_forward_add_f32(sgpt_tensor* src0, sgpt_tensor* src1, sgpt_tensor* dst) {
  static_assert(SGPT_MAX_DIMS == 4, "SPGT_MAX_DIMS != 4");
  assert(src0->ne[0] == dst->ne[0]);
  assert(src0->ne[1] == dst->ne[1]);
  assert(src0->ne[2] == dst->ne[2]);
  assert(src0->ne[3] == dst->ne[3]);
  assert(src1->ne[0] == dst->ne[0]);
  assert(src1->ne[1] == dst->ne[1]);
  assert(src1->ne[2] == dst->ne[2]);
  assert(src1->ne[3] == dst->ne[3]);
  for (int i3=0; i3<src0->ne[3]; i3++) {
    size_t loc3 = src0->nb[3] * i3;
    for (int i2=0; i2<src0->ne[2]; i2++) {
      size_t loc2 = loc3 + src0->nb[2] * i2;
      for (int i1=0; i1<src0->ne[1]; i1++) {
        size_t loc1 = loc2 + src0->nb[1] * i1;
        for (int i0=0; i0<src0->ne[0]; i0++) {
          size_t loc0 = loc1 + src0->nb[0] * i0;
          ((float*)(dst->data + loc0))[0] = ((float*)(src0->data + loc0))[0];
          ((float*)(dst->data + loc0))[0] += ((float*)(src1->data + loc0))[0];
        }
      }
    }
  }
}

static void sgpt_compute_forward_add_i32(sgpt_tensor* src0, sgpt_tensor* src1, sgpt_tensor* dst) {
  static_assert(SGPT_MAX_DIMS == 4, "SPGT_MAX_DIMS != 4");
  assert(src0->ne[0] == dst->ne[0]);
  assert(src0->ne[1] == dst->ne[1]);
  assert(src0->ne[2] == dst->ne[2]);
  assert(src0->ne[3] == dst->ne[3]);
  assert(src1->ne[0] == dst->ne[0]);
  assert(src1->ne[1] == dst->ne[1]);
  assert(src1->ne[2] == dst->ne[2]);
  assert(src1->ne[3] == dst->ne[3]);
  for (int i3=0; i3<src0->ne[3]; i3++) {
    size_t loc3 = src0->nb[3] * i3;
    for (int i2=0; i2<src0->ne[2]; i2++) {
      size_t loc2 = loc3 + src0->nb[2] * i2;
      for (int i1=0; i1<src0->ne[1]; i1++) {
        size_t loc1 = loc2 + src0->nb[1] * i1;
        for (int i0=0; i0<src0->ne[0]; i0++) {
          size_t loc0 = loc1 + src0->nb[0] * i0;
          ((int32_t*)(dst->data + loc0))[0] = ((int32_t*)(src0->data + loc0))[0];
          ((int32_t*)(dst->data + loc0))[0] += ((int32_t*)(src1->data + loc0))[0];
        }
      }
    }
  }
}

static void sgpt_compute_forward_add(sgpt_tensor* src0, sgpt_tensor* src1, sgpt_tensor* dst) {
  assert(src0->type == dst->type);
  assert(src1->type == dst->type);
  switch (src0->type) {
    case SGPT_TYPE_F32:
      sgpt_compute_forward_add_f32(src0, src1, dst);
      break;
    case SGPT_TYPE_I32:
      sgpt_compute_forward_add_i32(src0, src1, dst);
      break;
    default:
      assert(false);
  }
}

static void sgpt_compute_forward(sgpt_tensor* tensor) {
  switch (tensor->op) {
    case SGPT_OP_DUP:
      sgpt_compute_forward_dup(tensor->src0, tensor);
      break;
    case SGPT_OP_ADD:
      sgpt_compute_forward_add(tensor->src0, tensor->src1, tensor);
      break;
    case SGPT_OP_NONE:
      break;
    default:
      assert(false);
  }
}

void sgpt_graph_compute(sgpt_context* ctx, sgpt_cgraph* cgraph) {
  for (int i=0; i<cgraph->n_nodes; i++) {
    sgpt_compute_forward(cgraph->nodes[i]);
  }
}
