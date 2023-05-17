#include <stdio.h>

#include "sgpt.h"

int main(void) {
  uint8_t mem_buffer[1024];
  sgpt_context* ctx = sgpt_init((sgpt_init_params){
      .mem_size = 1024,
      .mem_buffer = (void*)mem_buffer,
  });

  sgpt_tensor* a = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 1);
  sgpt_tensor* b = sgpt_new_tensor_1d(ctx, SGPT_TYPE_I32, 1);
  sgpt_tensor* c = sgpt_add(ctx, a, b);

  sgpt_cgraph gf = sgpt_build_forward(c);
  sgpt_set_i32_1d(a, 0, 1);
  sgpt_set_i32_1d(b, 0, 2);
  sgpt_graph_compute(ctx, &gf);

  printf("1 + 2 = %d", sgpt_get_i32_1d(c, 0));
}
