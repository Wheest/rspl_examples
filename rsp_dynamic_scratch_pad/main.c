#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  Concat = 0x0,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPConcat(int16_t *a, int16_t *b, int16_t *dest, int sizeA,
                             int sizeB) {
  extern uint32_t vec_id;
  rspq_write(vec_id, Concat, PhysicalAddr(a), PhysicalAddr(b),
             PhysicalAddr(dest), sizeA * 2, sizeB * 2);
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
  int sizeA = 8 * 4;
  int sizeB = 8 * 2;
  int16_t *A = malloc_uncached_aligned(8, sizeof(int16_t) * sizeA);
  int16_t *B = malloc_uncached_aligned(8, sizeof(int16_t) * sizeB);
  int16_t *dest = malloc_uncached_aligned(8, sizeof(int16_t) * (sizeA + sizeB));

  /* printf("Data before RSP\n"); */
  int16_t val = 0;
  for (int i = 0; i < sizeA; i++) {
    A[i] = val++;
  }
  for (int i = 0; i < sizeB; i++) {
    B[i] = val++;
  }

  printf("\nTransfering data to RSP...\n");
  RSPConcat(A, B, dest, sizeA, sizeB);
  rspq_wait();
  printf("Done\n");

  printf("Data after RSP (should be sequential since we concat'd)\n");
  for (int i = 0; i < sizeA + sizeB; i++) {
    printf("%d ", dest[i]);
  }
  printf("\n");

  // Clean up
  vec_close();

  return 0;
}
