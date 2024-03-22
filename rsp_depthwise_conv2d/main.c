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
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

static inline void RSPDepthConv(int16_t *dest, int16_t *matA, int16_t *matB) {
  extern uint32_t vec_id;
  rspq_write(vec_id, MatMul8x8, PhysicalAddr(dest), PhysicalAddr(matA),
             PhysicalAddr(matB));
}

void sequential_depthwise_conv2d_simd(const int16_t *input_data,
                                      int16_t *output, const int16_t *weights,
                                      int input_height, int input_width,
                                      int input_depth, int kernel_height,
                                      int kernel_width, int output_height,
                                      int output_width, int stride,
                                      int padding) {
  int vector_size = 8; // Simulate a SIMD vector size of 8
  for (int h = 0; h < output_height; h++) {
    for (int w = 0; w < output_width; w++) {
      for (int c = 0; c < input_depth; c += vector_size) {
        int16_t vector_sum[8] = {0}; // Temporary sum for each vector

        // Loop over the kernel's height and width
        for (int i = 0; i < kernel_height; i++) {
          for (int j = 0; j < kernel_width; j++) {
            for (int v = 0; v < vector_size;
                 v++) { // This loop simulates the SIMD operation
              int input_h = h * stride + i - padding;
              int input_w = w * stride + j - padding;
              if (input_h >= 0 && input_w >= 0 && input_h < input_height &&
                  input_w < input_width) {
                vector_sum[v] +=
                    input_data[(input_h * input_width + input_w) * input_depth +
                               c + v] *
                    weights[(i * kernel_width + j) * input_depth + c + v];
              }
            }
          }
        }

        for (int v = 0; v < vector_size; v++) {
          output[(h * output_width + w) * input_depth + c + v] = vector_sum[v];
        }
      }
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

  printf("Allocating arrays for int16 depthwise convolution\n");
  int input_height = 4;
  int input_width = 4;
  int input_depth = 8; // Depth is set to the SIMD vector size
  int kernel_height = 3;
  int kernel_width = 3;
  int output_height = 2;
  int output_width = 2;
  int stride = 1;
  int padding = 0;

  // Allocate and initialize input, output and weights
  int16_t *input_data = malloc_uncached_aligned(
      8, input_height * input_width * input_depth * sizeof(int16_t));
  int16_t *output = malloc_uncached_aligned(
      8, output_height * output_width * input_depth * sizeof(int16_t));
  int16_t *output_target = malloc_uncached_aligned(
      8, output_height * output_width * input_depth * sizeof(int16_t));
  int16_t *weights = malloc_uncached_aligned(
      8, kernel_height * kernel_width * input_depth * sizeof(int16_t));

  // Initialize input data and weights with sequential values for the example
  int16_t val = 0;
  int16_t direction = 1;
  for (int i = 0; i < input_height * input_width * input_depth; i++) {
    input_data[i] = val;
    val += direction;
    if (i % 7 == 6)
      direction = -direction;
  }

  val = 0;
  direction = 1;
  for (int i = 0; i < kernel_height * kernel_width * input_depth; i++) {
    weights[i] = val;
    val += direction;
    if (i % 8 == 7) {
      direction = -direction;
    }
  }

  /* printf("X\n"); */
  /* for (int c = 0; c < 8; ++c) { */
  /*   printf("%d ", input_data[c]); */
  /* } */

  /* printf("\n"); */

  /* printf("W\n"); */
  /* for (int c = 0; c < 8; ++c) { */
  /*   printf("%d ", weights[c]); */
  /* } */
  /* printf("\n"); */

  unsigned long start = get_ticks();
  // Perform the convolution
  sequential_depthwise_conv2d_simd(input_data, output_target, weights,
                                   input_height, input_width, input_depth,
                                   kernel_height, kernel_width, output_height,
                                   output_width, stride, padding);
  unsigned long time_spent = get_ticks() - start;
  printf("\nCPU baseline time (CPU ticks): %lu\n", time_spent);

  // Print the output
  printf("Target\n");
  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      printf("Target[%d, %d]: ", i, j);
      for (int k = 0; k < input_depth; k++) {
        printf("%d ", output_target[(i * output_width + j) * input_depth + k]);
      }
      printf("\n");
    }
  }

  rspq_wait();
  start = get_ticks();
  RSPDepthConv(output, input_data, weights);
  rspq_wait();
  time_spent = get_ticks() - start;
  printf("\nRSP time (CPU ticks): %lu\n", time_spent);
  printf("Output\n");
  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      printf("Output[%d, %d]: ", i, j);
      for (int k = 0; k < input_depth; k++) {
        printf("%d ", output[(i * output_width + j) * input_depth + k]);
      }
      printf("\n");
    }
  }

  // Compare the results
  int differences = 0;
  for (int i = 0; i < output_height * output_width * input_depth; ++i) {
    if (output[i] != output_target[i]) {
      printf("Difference found at index %d: %d (RSP) vs %d (C)\n", i, output[i],
             output_target[i]);
      differences++;
    }
  }

  if (differences == 0) {
    printf("The RSP computation is correct.\n");
  } else {
    printf("There are %d differences. The RSP computation is not correct.\n",
           differences);
  }

  // Clean up
  vec_close();

  return 0;
}
