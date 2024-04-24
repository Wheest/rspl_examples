#include "utils.h"
#include <libdragon.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1

#define MIN(a, b) ((a) < (b) ? (a) : (b))
DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

void copy_slice_to_full_with_stride(int32_t *dest, int32_t *output_pad_part,
                                    int output_partition_height, int out_w,
                                    int slice_count, int in_depth,
                                    int depth_slice_num, size_t slice_depth,
                                    int max_output_parition_height) {
  // Copies a slice into the final destination with the appripriate stride
  // e.g., a slice of size 4x4x8 could be copied into a 4x4x16 destination
  // slice_depth should probably be the inner loop size (e.g., 8)
  /* printf("Copying slice %d\n", slice_count); */
  for (int y = 0; y < max_output_parition_height; y++) {
    for (int x = 0; x < out_w; x++) {
      // Calculate the initial destination address with offsets and strided gaps
      int32_t *dest_ptr =
          &dest[slice_count * output_partition_height * out_w * in_depth +
                (y * out_w + x) * in_depth + (depth_slice_num * slice_depth)];

      int offset = y * out_w * slice_depth + x * slice_depth;
      uint16_t *src = (uint16_t *)(output_pad_part + offset);
      for (int c = 0; c < slice_depth; c++) {
        // Perform the copy operation with stride (and reconstruct the 32-bit
        // values)
        dest_ptr[c] = ((uint32_t)src[c] << 16) | src[c + 8];
      }
    }
  }
}

void offline_weight_reshape(const int8_t *kernels, int8_t *kernel_reshape,
                            int kdim_h, int kdim_w, int out_c,
                            int split_factor) {
  int split_size = kdim_h * kdim_w * split_factor;
  int num_splits = out_c / split_factor;

  for (int split = 0; split < num_splits; ++split) {
    for (int row = 0; row < kdim_h; ++row) {
      for (int col = 0; col < kdim_w; ++col) {
        for (int ch = 0; ch < split_factor; ++ch) {
          int src_idx = (row * kdim_w * out_c) + (col * out_c) +
                        (split * split_factor) + ch;
          int dest_idx = (split * split_size) + (row * kdim_w * split_factor) +
                         (col * split_factor) + ch;

          kernel_reshape[dest_idx] = kernels[src_idx];
        }
      }
    }
  }
}

enum {
  DMAWeights = 0x0,
  DMAInputs = 0x1,
  DepthConv = 0x2,
  SetArgs = 0x3,
  DMAOutWeights = 0x3,
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

void generate_padded_slices_with_depth_slice(
    int16_t *data, int16_t *padded_input_partition, int start_h, int in_h,
    int in_w, int in_depth, int max_slice_height, int padding, int overlap,
    int depth_slice // New parameter indicating which set of 8 channels to copy
) {
  // Calculate the size of the padded area (height)
  int len = MIN(start_h + max_slice_height, in_h) - start_h;

  // Calculate depth start and end based on the depth_slice
  int depth_start =
      depth_slice * 8; // Start at the depth_slice set of 8 channels
  int depth_end =
      MIN(depth_start + 8, in_depth); // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0,
         sizeof(int16_t) * max_slice_height * (in_w + 2 * padding) * 8);

  // Determine slice dimensions
  int actual_start = (start_h == 0) ? start_h : start_h - padding;
  int actual_end = start_h + len;

  int target_start_h = (start_h == 0) ? padding : 0;

  // Copy data to the padded slice
  for (int h = actual_start; h < actual_end; ++h) {
    for (int w = 0; w < in_w; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        int16_t value = data[(h * in_w * in_depth) + (w * in_depth) + d];
        int target_row = target_start_h + h - actual_start;
        int target_col = w + padding;
        int target_depth = d - depth_start;

        padded_input_partition[(target_row * (in_w + 2 * padding) * 8) +
                               (target_col * 8) + target_depth] = value;
      }
    }
  }
}

static inline void
RSPDepthConvTiledPadded(int32_t *dest, int16_t *input_data, int8_t *weights,
                        int input_partition_height, int in_h, int in_w,
                        int in_c, int out_h, int out_w, int k_h, int k_w,
                        int output_partition_height, int padding, int stride) {
  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  /* printf("Weights reshape\n"); */
  /* printInt8ArrayHWC(weights, 3, 3, 8); */

  const int wbytes = sizeof(int8_t);
  const int in_bytes = sizeof(int16_t);
  const int out_bytes = sizeof(int32_t);
  int16_t *input_pad_part = malloc_uncached_aligned(
      8, input_partition_height * (in_w + 2 * padding) * 8 * in_bytes);
  int32_t *output_pad_part = malloc_uncached_aligned(
      8, output_partition_height * out_w * 8 * out_bytes);
  const int out_part_size = output_partition_height * out_w * 8;
  const int in_part_size =
      input_partition_height * (in_w + 2 * padding) * 8 * in_bytes;
  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  const int w_stride_slice = 8 * in_bytes * stride;
  const int w_slide_byte_offset = 8 * in_bytes * (in_w + 2 * padding);
  const int h_slide_byte_offset =
      (k_w * 8 * in_bytes) - 16 +
      (stride - 1) * 8 * (in_h + 2 * padding) * in_bytes;

  rspq_write(vec_id, SetArgs, output_partition_height, out_w, w_stride_slice,
             w_slide_byte_offset, h_slide_byte_offset, w_part_size,
             in_part_size);

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int slice_count = 0;
    int remaining_out_values = out_h * out_w * 8; // Remaining values to copy
    int max_output_partition_height = output_partition_height;

    // Copy weights to the RSP once
    rspq_write(vec_id, DMAWeights,
               PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    // Generate first slice
    generate_padded_slices_with_depth_slice(
        input_data, (int16_t *)input_pad_part, 0, in_h, in_w, in_c,
        input_partition_height, padding, overlap, depth_slice);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    int start_h_next = input_partition_height - overlap;

    for (int start_h = 0; start_h < in_h;
         start_h += input_partition_height - overlap) {
      rspq_wait();

      // Process the padded partition on the RSP
      rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part),
                 out_part_size * out_bytes);
      // Generate next slice while the current one is being
      // processed
      if (start_h_next <= in_h) {
        generate_padded_slices_with_depth_slice(
            input_data, input_pad_part, start_h_next, in_h, in_w, in_c,
            input_partition_height, padding, overlap, depth_slice);
        start_h_next += input_partition_height - overlap;
      }

      // Copy back the processed partition to the final outputs,
      // DMA the new input slice to the RSP at the same time
      rspq_wait();
      rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

      if (remaining_out_values < out_part_size) {
        // Cover the case where our final output slice is larger than required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }

      /* printf("Output slice %d: \n", slice_count); */
      /* debug_hexdump(output_pad_part, 32 * sizeof(int32_t)); */
      copy_slice_to_full_with_stride(
          dest, output_pad_part, output_partition_height, out_w, slice_count,
          in_c, depth_slice, 8, max_output_partition_height);
      remaining_out_values -= output_partition_height * out_w * 8;
      /* printf("Post reshape %d: \n", slice_count); */
      /* debug_hexdump(dest, 32 * sizeof(int32_t)); */
      // debug_hexdump(dest, 16 * sizeof(int32_t));

      /* printf("Output slice %d: \n", slice_count); */
      /* printInt32ArrayHWC(output_pad_part, output_partition_height, out_h, 8);
       */

      /* if (slice_count > 0) */
      /*   break; */
      /* if (slice_count > 1) */
      /*   break; */
      /* if (depth_slice > 0 && slice_count >= 0) */
      /*   break; */
      slice_count++;
      /* break; */
    }
    /* break; */
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

  int input_height = 16;
  int input_width = 16;
  int input_depth = 16;
  int kernel_height = 3;
  int kernel_width = 3;
  /* int output_height = 2; */
  /* int output_width = 2; */
  int stride = 1;
  int padding = 1;
  int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
  int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
  int input_partition_height = 4;
  int output_partition_height = 2;

  printf("Allocating arrays for int16 depthwise convolution\n");
  printf("Input dimensions: %d x %d x %d\n", input_height, input_width,
         input_depth);
  printf("Padding: %d, stride: %d\n", padding, stride);
  printf("Kernel dimensions: %d x %d\n", kernel_height, kernel_width);
  printf("Output shape: %d x %d x %d\n", output_height, output_width,
         input_depth);

  // Allocate and initialize input, output and weights
  const int wbytes = sizeof(int8_t);
  const int in_bytes = sizeof(int16_t);
  const int out_bytes = sizeof(int32_t);
  int16_t *input_data = malloc_uncached_aligned(8, input_height * input_width *
                                                       input_depth * in_bytes);
  int32_t *output = malloc_uncached_aligned(8, output_height * output_width *
                                                   input_depth * out_bytes);
  int32_t *output_target = malloc_uncached_aligned(
      8, output_height * output_width * input_depth * out_bytes);
  int8_t *weights = malloc_uncached_aligned(8, kernel_height * kernel_width *
                                                   input_depth * wbytes);

  // Initialize input data and weights with sequential values for the example
  int16_t val = 1024;
  int16_t direction = 1;
  for (int i = 0; i < input_height * input_width * input_depth; i++) {
    input_data[i] = val;
    val += direction;
    if (i % 4096 == 4095)
      direction = -direction;
  }

  int8_t val2 = -120;
  int8_t direction2 = 1;
  for (int i = 0; i < kernel_height * kernel_width * input_depth; i++) {
    weights[i] = val2;
    val2 += direction2;
    if (i % 8 == 7) {
      direction2 = -direction2;
    }
  }

  /* printf("Weights\n"); */
  /* printInt8ArrayHWC(weights, kernel_height, kernel_width, input_depth); */

  int8_t *weights_reshape = malloc_uncached_aligned(
      8, kernel_height * kernel_width * input_depth * wbytes);
  offline_weight_reshape(weights, weights_reshape, kernel_height, kernel_width,
                         input_depth, 8);

  /* printf("Weights reshape\n"); */
  /* printIntArrayHWC(weights_reshape, kernel_height, kernel_width,
   * input_depth); */
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

  rspq_wait();
  start = get_ticks();
  RSPDepthConvTiledPadded(
      output, input_data, weights_reshape, input_partition_height, input_height,
      input_width, input_depth, output_height, output_width, kernel_height,
      kernel_width, output_partition_height, padding, stride);
  rspq_wait();
  unsigned long time_spent_rsp = get_ticks() - start;
  printf("\nRSP time (CPU ticks): %lu\n", time_spent_rsp);

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

  /* printf("Output\n"); */
  /* printInt32ArrayHWC(output, output_height, output_width, input_depth); */
  /* debug_hexdump(output, 64 * sizeof(int32_t)); */

  /* printf("Output_target\n"); */
  /* printInt32ArrayHWC(output_target, output_height, output_width,
   * input_depth); */
  /* debug_hexdump(output_target, 64 * sizeof(int32_t)); */

  if (differences == 0) {
    printf("The RSP computation is correct (speedup %f).\n",
           (float)time_spent_cpu / (float)time_spent_rsp);
  } else {
    printf("There are %d differences. The RSP computation is not correct.\n",
           differences);
  }

  // Clean up
  vec_close();

  return 0;
}
