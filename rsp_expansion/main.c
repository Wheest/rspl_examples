#include <inttypes.h> // for PRIx32 macro
#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  Shift = 0x0,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPShift(int16_t *dest, int8_t *a) {
  extern uint32_t vec_id;
  rspq_write(vec_id, Shift, PhysicalAddr(dest), PhysicalAddr(a));
}

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();

  vec_init();
  printf("Init'd RSP overlay\n");
  printf("This program adds 300 to the values, which are int8.\nThe RSP will "
         "shift the values to int16\n");

  // allocate and copy over data to the RSP
  int16_t *output_array = malloc_uncached_aligned(8, sizeof(int16_t) * 16);
  int8_t *a = malloc_uncached_aligned(8, sizeof(int8_t) * 16);

  printf("Data before RSP\n");
  for (int i = 0; i < 16; i++) {
    a[i] = 50 + i;
    printf("%d ", a[i]);
  }

  printf("\nTransfering data to RSP...\n");
  RSPShift(output_array, a);
  rspq_wait();
  printf("Done\n");

  printf("Data after RSP\n");
  for (int i = 0; i < 16; i++) {
    printf("%d ", output_array[i]);
  }

  char *correct = "aye";
  for (int i = 0; i < 8; i++) {
    if (output_array[i] != (int32_t)a[i] + 300) {
      correct = "naw";
    }
  }
  printf("\n");
  printf("Correct: %s\n", correct);

  // Clean up
  vec_close();

  return 0;
}
