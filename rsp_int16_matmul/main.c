#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1

DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

enum {
  MatMul8x8 = 0x0,
};

void vec_init() {
  rspq_init();

  // Initialize the saved state
  void *state = UncachedAddr(rspq_overlay_get_state(&rsp_simple));
  memset(state, 0, 0x400);

  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPMatMul8x8(int16_t *dest, int16_t *matA, int16_t *matB) {
  extern uint32_t vec_id;
  rspq_write(vec_id, MatMul8x8, PhysicalAddr(dest), PhysicalAddr(matA),
             PhysicalAddr(matB));
}

void MatMulCPU(int16_t *dest, const int16_t *matA, const int16_t *matB, int n) {
  // CPU matmul, with row-major A and transposed B
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int16_t sum = 0;
      for (int k = 0; k < n; ++k) {
        sum += matA[r * n + k] * matB[c * n + k];
      }
      dest[r * n + c] = sum;
    }
  }
}

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();

  // Initialize the "vec" library (see vec.h)
  vec_init();
  printf("Init'd RSP overlay\n");

  int16_t *output_matrix = malloc_uncached_aligned(8, sizeof(int16_t) * 8 * 8);
  int16_t *matrix_A = malloc_uncached_aligned(8, sizeof(int16_t) * 8 * 8);
  int16_t *matrix_B = malloc_uncached_aligned(8, sizeof(int16_t) * 8 * 8);
  int16_t *output_matrix_target =
      malloc_uncached_aligned(8, sizeof(int16_t) * 8 * 8);

  printf("A\n");
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      int index = r * 8 + c;
      matrix_A[index] = index;
      printf("%d ", matrix_A[index]);
    }
    printf("\n");
  }
  printf("\n");

  printf("B\n");
  for (int c = 0; c < 8; ++c) {
    for (int r = 0; r < 8; ++r) {
      int index = c * 8 + r;
      matrix_B[index] = index;
      printf("%d ", matrix_B[index]);
    }
    printf("\n");
  }
  printf("\n");
  MatMulCPU(output_matrix_target, matrix_A, matrix_B, 8);

  printf("Target\n");
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      int index = r * 8 + c;
      printf("%d ", output_matrix_target[index]);
    }
    printf("\n");
  }
  printf("\n");

  printf("\nTransfering data to RSP...\n");
  RSPMatMul8x8(output_matrix, matrix_A, matrix_B);
  rspq_wait();
  printf("Done\n");

  printf("Data after RSP\n");
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      int index = r * 8 + c;
      printf("%d ", output_matrix[index]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Target\n");
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      int index = r * 8 + c;
      printf("%d ", output_matrix_target[index]);
    }
    printf("\n");
  }
  printf("\n");

  // Compare the results
  int differences = 0;
  for (int i = 0; i < 64; ++i) {
    if (output_matrix[i] != output_matrix_target[i]) {
      printf("Difference found at index %d: %d (RSP) vs %d (C)\n", i,
             output_matrix[i], output_matrix_target[i]);
      differences++;
    }
  }

  if (differences == 0) {
    printf("All values are identical. The RSP computation is correct.\n");
  } else {
    printf("There are %d differences. The RSP computation is not correct.\n",
           differences);
  }

  // Clean up
  vec_close();

  return 0;
}
