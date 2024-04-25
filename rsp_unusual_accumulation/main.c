#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  ComputeWrong = 0x0,
  ComputeCorrect = 0x1,
};

void vec_init() {
  rspq_init();
  vec_id = rspq_overlay_register(&rsp_simple);
}

void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPComputeWrong(int32_t *dest, int8_t *x, int8_t *w) {
  extern uint32_t vec_id;
  rspq_write(vec_id, ComputeWrong, PhysicalAddr(dest), PhysicalAddr(x),
             PhysicalAddr(w));
}

static inline void RSPComputeCorrect(int32_t *dest, int8_t *x, int8_t *w) {
  extern uint32_t vec_id;
  rspq_write(vec_id, ComputeCorrect, PhysicalAddr(dest), PhysicalAddr(x),
             PhysicalAddr(w));
}

void reconstruct_vector(int32_t *src, int32_t *dst, size_t size) {
  uint16_t *out = (uint16_t *)src;
  for (int i = 0; i < 8; i++) {
    dst[i] = ((uint32_t)out[i] << 16) | out[i + 8];
  }
}

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();

  vec_init();
  printf("Init'd RSP overlay\n");

  // allocate and copy over data to the RSP
  int num_elems = 16;
  int8_t *x = malloc_uncached_aligned(8, sizeof(int8_t) * num_elems);
  int8_t *w = malloc_uncached_aligned(8, sizeof(int8_t) * num_elems);
  int32_t *y = malloc_uncached_aligned(8, sizeof(int32_t) * num_elems);

  // Initialize data
  for (int i = 0; i < num_elems; i++) {
    x[i] = 0;
    w[i] = 0;
  }
  x[0] = 2;
  w[0] = 3;
  x[8] = 4;
  w[8] = 5;

  printf("\nComputing...\n");
  RSPComputeWrong(y, x, w);
  rspq_wait();
  printf("Done\n");

  printf("X:\n");
  for (int i = 0; i < num_elems; i++) {
    printf("%d, ", x[i]);
  }

  printf("\nW:\n");
  for (int i = 0; i < num_elems; i++) {
    printf("%d, ", w[i]);
  }

  int32_t y_reconstructed[num_elems];
  reconstruct_vector(y, y_reconstructed, num_elems);

  // Target value
  int32_t target = (int32_t)w[0] * (int32_t)x[0];
  target += (int32_t)w[8] * (int32_t)x[8];

  printf("\n\nComputing\t y = (w[0] * x[0]) \t+ (w[8] * x[8])\n");
  printf("\t\t\t y = (%d * %d) \t\t+ (%d * %d)\n", w[0], x[0], w[8], x[8]);

  printf("\n\nWrong version:\n");
  printf("Target is: %ld\n", target);
  printf("Actual is: %ld\n", y_reconstructed[0]);

  printf("\nCorrect version:\n");

  RSPComputeCorrect(y, x, w);
  rspq_wait();

  reconstruct_vector(y, y_reconstructed, num_elems);
  printf("Target is: %ld\n", target);
  printf("Actual is: %ld\n", y_reconstructed[0]);

  // Clean up
  vec_close();

  return 0;
}
