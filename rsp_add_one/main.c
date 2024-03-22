#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  MatCopyIn = 0x0,
  MatCopyOut = 0x1,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void transfer(int16_t *dest) {
  extern uint32_t vec_id;
  rspq_write(vec_id, MatCopyIn, PhysicalAddr(dest));
}
static inline void transfer_out(int16_t *dest) {
  extern uint32_t vec_id;
  rspq_write(vec_id, MatCopyOut, PhysicalAddr(dest));
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
  int16_t *output_matrix = malloc_uncached_aligned(8, sizeof(int16_t) * 8);

  printf("Data before RSP\n");
  for (int i = 0; i < 8; i++) {
    output_matrix[i] = i;
    printf("%d ", output_matrix[i]);
  }
  printf("\nTransfering data to RSP...\n");
  transfer(output_matrix);
  rspq_wait();
  printf("Done\n");

  transfer_out(output_matrix);
  rspq_wait();
  printf("Data after RSP (should have added 1)\n");
  for (int i = 0; i < 8; i++) {
    printf("%d ", output_matrix[i]);
  }
  printf("\n");

  // Clean up
  vec_close();

  return 0;
}
