#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1

#define MIN(a, b) ((a) < (b) ? (a) : (b))
DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

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

void printIntArrayHWC(int16_t *array, int height, int width, int in_c) {
  printf("[");
  for (int h = 0; h < height; ++h) {
    // Check for first line to avoid leading comma for rows
    if (h > 0) {
      printf(",\n ");
    }
    printf("[");
    for (int w = 0; w < width; ++w) {
      // Check for first column to avoid leading comma for columns
      if (w > 0) {
        printf(", ");
      }
      printf("[");
      for (int c = 0; c < in_c; ++c) {
        // Retrieve the value at [h, w, c]
        int16_t val = array[(h * width * in_c) + (w * in_c) + c];
        if (c > 0) {
          printf(", ");
        }
        printf("%d", val);
      }
      printf("]");
    }
    printf("]");
  }
  printf("]\n");
}

enum {
  DMAWeights = 0x0,
  DMAInputs = 0x1,
  DepthConv = 0x2,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

void generate_padded_slices(int16_t *data, int16_t *padded_input_partition,
                            int start_h, int in_h, int in_w, int in_depth,
                            int max_slice_height, int padding, int overlap) {
  // Calculate the size of the padded area (height)
  int len =
      (start_h + max_slice_height > in_h) ? in_h - start_h : max_slice_height;

  // Clear the slice area
  memset(padded_input_partition, 0,
         sizeof(int16_t) * (max_slice_height) * (in_w + 2 * padding) *
             in_depth);

  // Determine slice dimensions
  int actual_start = (start_h == 0) ? start_h : start_h - padding;
  int actual_end = start_h + len;

  int target_start = (start_h == 0) ? padding : 0;

  // Copy data to the padded slice
  for (int h = actual_start; h < actual_end; ++h) {
    for (int w = 0; w < in_w; ++w) {
      for (int d = 0; d < in_depth; ++d) {
        int16_t value = *((data + h * in_w * in_depth) + w * in_depth + d);
        int target_row = target_start + h - actual_start;
        int target_col = w + padding;

        *((padded_input_partition +
           target_row * (in_w + 2 * padding) * in_depth) +
          target_col * in_depth + d) = value;
      }
    }
  }
}

static inline void
RSPDepthConvTiledPadded(int16_t *dest, int16_t *input_data, int16_t *weights,
                        int input_partition_height, int in_h, int in_w,
                        int in_c, int out_h, int out_w,
                        int output_partition_height, int padding) {
  extern uint32_t vec_id;
  // Copy weights to the RSP once
  rspq_write(vec_id, DMAWeights, PhysicalAddr(weights));

  // raise not implemented error if in_c is not 8
  if (in_c != 8) {
    printf("Error: in_c must be 8\n");
    return;
  }

  /* printf("Weights\n"); */
  /* printIntArrayHWC(weights, 3, 3, 8); */

  int16_t *input_pad_part = malloc_uncached_aligned(
      8, input_partition_height * (in_w + 2 * padding) * 8 * sizeof(int16_t));
  int16_t *output_pad_part = malloc_uncached_aligned(
      8, output_partition_height * out_w * 8 * sizeof(int16_t));

  int slice_count = 0;
  int overlap = 2;

  // Generate first slice
  generate_padded_slices(input_data, (int16_t *)input_pad_part, 0, in_h, in_w,
                         in_c, input_partition_height, padding, overlap);
  rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part));
  int start_h_next = input_partition_height - overlap;

  int remaining_out_values = out_h * out_w * in_c;
  for (int start_h = 0; start_h <= in_h;
       start_h += input_partition_height - overlap) {
    /* generate_padded_slices(input_data, (int16_t *)input_pad_part, start_h,
     * in_h, */
    /*                        in_w, in_c, input_partition_height, padding, */
    /*                        overlap); */
    /* printIntArrayHWC(input_pad_part, 4, 10, 8); */
    /* rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part)); */
    /* dma_wait(); */
    rspq_wait();
    // Process the padded partition on the RSP
    rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part));
    // Generate next slice while the current one is being processed
    if (start_h_next <= in_h) {
      generate_padded_slices(input_data, (int16_t *)input_pad_part,
                             start_h_next, in_h, in_w, in_c,
                             input_partition_height, padding, overlap);
      start_h_next += input_partition_height - overlap;
    }

    // Copy back the processed partition to the final outputs, DMA the new input
    // slice to the RSP at the same time
    rspq_wait();
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part));
    /* printf("Input slice %d: \n", slice_count); */
    /* printIntArrayHWC(input_pad_part, input_partition_height, */
    /*                  (in_w + 2 * padding), 8); */
    memcpy(&dest[slice_count * (output_partition_height * out_w * 8)],
           output_pad_part,
           MIN(remaining_out_values * sizeof(int16_t),
               output_partition_height * out_w * 8 * sizeof(int16_t)));
    remaining_out_values -= output_partition_height * out_w * 8;
    /* printf("Output slice %d: \n", slice_count); */
    /* printIntArrayHWC(output_pad_part, output_partition_height, out_h, out_w);
     */

    /* if (slice_count > 0) */
    /*   break; */
    slice_count++;
    /* break; */
  }
}
/* for (int i = 0; i < num_partitions; i++) */
/*   rspq_write( */
/*       vec_id, MatMul8x8, */
/*       PhysicalAddr( */
/*           &dest[i * (output_height * output_width * 8 / num_partitions)]),
 */
/*       PhysicalAddr( */
/*           &input_data[i * (input_height * input_width * 8 /
 * num_partitions)]), */
/*       PhysicalAddr(weights)); */
/* } */

int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();

  // Initialize the "vec" library (see vec.h)
  vec_init();
  printf("Init'd RSP overlay\n");

  /* int input_height = 4; */
  /* int input_width = 4; */
  /* int input_height = 8; */
  /* int input_width = 8; */
  int input_height = 16;
  int input_width = 16;
  int input_depth = 8; // Depth is set to the SIMD vector size
  int kernel_height = 3;
  int kernel_width = 3;
  /* int output_height = 2; */
  /* int output_width = 2; */
  int stride = 1;
  int padding = 1;
  int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
  int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
  int input_partition_height = 5;
  int output_partition_height = 3;

  printf("Allocating arrays for int16 depthwise convolution\n");
  printf("Input dimensions: %d x %d x %d\n", input_height, input_width,
         input_depth);
  printf("Kernel dimensions: %d x %d\n", kernel_height, kernel_width);
  printf("Output shape: %d x %d x %d\n", output_height, output_width,
         input_depth);

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

  /* printf("Input\n"); */
  /* printIntArrayHWC(input_data, input_height, input_width, input_depth); */

  unsigned long start = get_ticks();
  // Perform the convolution
  sequential_depthwise_conv2d_simd(input_data, output_target, weights,
                                   input_height, input_width, input_depth,
                                   kernel_height, kernel_width, output_height,
                                   output_width, stride, padding);
  unsigned long time_spent_cpu = get_ticks() - start;
  printf("\nCPU baseline time (CPU ticks): %lu\n", time_spent_cpu);

  // Print the output
  /* printf("Target\n"); */
  /* int max_i = 4; */
  /* for (int i = 0; i < max_i; i++) { */

  /*   for (int j = 0; j < output_width; j++) { */
  /*     printf("Target[%d, %d]: ", i, j); */
  /*     for (int k = 0; k < input_depth; k++) { */
  /*       printf("%d ", output_target[(i * output_width + j) * input_depth +
   * k]); */
  /*     } */
  /*     printf("\n"); */
  /*   } */
  /* } */

  rspq_wait();
  start = get_ticks();
  RSPDepthConvTiledPadded(output, input_data, weights, input_partition_height,
                          input_height, input_width, input_depth, output_height,
                          output_width, output_partition_height, padding);
  rspq_wait();
  unsigned long time_spent_rsp = get_ticks() - start;
  printf("\nRSP time (CPU ticks): %lu\n", time_spent_rsp);

  /* printf("Output (output_width: %d)\n", output_width); */
  /* for (int i = 0; i < max_i; i++) { */
  /*   for (int j = 0; j < output_width; j++) { */
  /*     printf("Output[%d, %d]: ", i, j); */
  /*     for (int k = 0; k < input_depth; k++) { */
  /*       printf("%d ", output[(i * output_width + j) * input_depth + k]); */
  /*     } */
  /*     printf("\n"); */
  /*   } */
  /* } */

  // Compare the results
  int differences = 0;
  for (int i = 0; i < output_height * output_width * input_depth; ++i) {
    if (output[i] != output_target[i]) {
      /* printf("Difference found at index %d: %d (RSP) vs %d (C)\n", i,
       * output[i], */
      /*        output_target[i]); */
      differences++;
    }
  }

  if (differences == 0) {
    printf("The RSP computation is correct (speedup %f).\n",
           (float)time_spent_cpu / (float)time_spent_rsp);
  } else {
    printf("There are %d differences. The RSP computation is not correct.\n",
           differences);
  }

  /* printf("Output\n"); */
  /* printIntArrayHWC(output, output_height, output_width, input_depth); */

  /* printf("Output_target\n"); */
  /* printIntArrayHWC(output_target, output_height, output_width, input_depth);
   */

  // Clean up
  vec_close();

  return 0;
}
