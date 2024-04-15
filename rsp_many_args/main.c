#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define RSPQ_DATA_ADDRESS 32

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  SetArgs = 0x0,
  SumArgs = 0x1,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void set_args(int32_t a, int32_t b, int32_t c, int32_t d,
                            int32_t e) {
  extern uint32_t vec_id;
  rspq_write(vec_id, SetArgs, a, b, c, d, e);
}
static inline void sum_args(int32_t *dest) {
  extern uint32_t vec_id;
  rspq_write(vec_id, SumArgs, PhysicalAddr(dest));
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
  int32_t a = 1;
  int32_t b = 2;
  int32_t c = 3;
  int32_t d = 4;
  int32_t e = 5;
  int32_t target = a + b + c + d + e;
  int32_t *dest = malloc_uncached_aligned(1, sizeof(int32_t));

  set_args(a, b, c, d, e);

  sum_args(dest);
  rspq_wait();

  printf(
      "Data after RSP (should be sum of a:%ld, b:%ld, c:%ld, d:%ld, e:%ld)\n",
      a, b, c, d, e);
  printf("%ld\n", dest[0]);
  char *correct = "aye";
  if (dest[0] != target) {
    correct = "naw";
  }
  printf("Correct: %s\n", correct);

  // Clean up
  vec_close();

  return 0;
}
