#include <inttypes.h> // for PRIx32 macro
#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  VecMul = 0x0,
};

void reconstruct_vector(int32_t *src, int32_t *dst, size_t size) {
  uint16_t *out = (uint16_t *)src;
  for (int i = 0; i < 8; i++) {
    dst[i] = ((uint32_t)out[i] << 16) | out[i + 8];
  }
}

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPVecMul(int32_t *dest, int16_t *a, int16_t *b) {
  extern uint32_t vec_id;
  rspq_write(vec_id, VecMul, PhysicalAddr(dest), PhysicalAddr(a),
             PhysicalAddr(b));
}

void printi16ArrayAsHex(int16_t *array, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    // Use PRIx16 to correctly format int16_t value as hex
    printf("%04" PRIx16 " ", (uint16_t)array[i]);
  }
  printf("\n");
}

void printi32ArrayAsHex(int32_t *array, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    // Correctly handling int32_t with casting and using PRIx32 macro
    printf("%08" PRIx32 " ", (uint32_t)array[i]);
  }
  printf("\n");
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
  int32_t *output_array = malloc_uncached_aligned(8, sizeof(int32_t) * 8);
  int16_t *a = malloc_uncached_aligned(8, sizeof(int16_t) * 8);
  int16_t *b = malloc_uncached_aligned(8, sizeof(int16_t) * 8);

  printf("Data before RSP\n");
  for (int i = 0; i < 8; i++) {
    a[i] = 50 + i;
    // set b to maximum int16_t value subtract i
    b[i] = 0x7FFF - i;
  }

  printf("\nTransfering data to RSP...\n");
  RSPVecMul(output_array, a, b);
  rspq_wait();
  printf("Done\n");

  printf("Data after RSP\n");
  for (int i = 0; i < 8; i++) {
    printf("%ld ", output_array[i]);
  }

  int32_t *output_array2 = malloc_uncached_aligned(8, sizeof(int32_t) * 8);

  // Reconstruct the vector, since right now it's in a format where
  // the first 4 elements are the upper 2 bytes of the 8 elements
  // and the last 4 elements are the lower 2 bytes of the 8 elements
  reconstruct_vector(output_array, output_array2, 8);

  printf("\nReconstructed\n");
  for (int i = 0; i < 8; i++) {
    printf("%ld ", output_array2[i]);
  }

  printf("\nShould be:\n");
  bool correct = true;
  for (int i = 0; i < 8; i++) {
    printf("%ld ", (int32_t)a[i] * b[i]);
    if (output_array2[i] != (int32_t)a[i] * b[i]) {
      correct = false;
    }
  }
  printf("\n");
  printf("Correct: %d\n", correct);

  // Clean up
  vec_close();

  return 0;
}
