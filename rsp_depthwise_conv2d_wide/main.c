#include "utils.h"
#include <libdragon.h>
#include <math.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1

#define MIN(a, b) ((a) < (b) ? (a) : (b))
DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

void copy_output_slice_to_full(int32_t *dest, const int32_t *output_pad_part,
                               const int output_partition_height,
                               const int output_partition_width,
                               const int out_w, const int h_slice_num,
                               const int w_slice_num, const int in_depth,
                               const int depth_slice_num,
                               const int max_output_partition_height) {
  /*
  Copies a slice into the final destination with the appropriate stride
  using for-loops.
  */
  const int slice_depth = 8; // 8 channels per slice
  for (int y = 0; y < max_output_partition_height; y++) {
    for (int x = 0; x < output_partition_width; x++) {
      // Calculate the initial destination index with offsets and strided gaps
      int dest_idx_y = h_slice_num * output_partition_height + y;
      int dest_idx_x = w_slice_num * output_partition_width + x;

      int32_t *dest_ptr =
          &dest[(dest_idx_y * out_w * in_depth) + (dest_idx_x * in_depth) +
                (depth_slice_num * slice_depth)];
      // cast to uint16_t as the RSP puts the upper 16 bits of all 8 elements in
      // the first 64 bits and the lower 16 bits in the second 64 bits
      uint16_t *src_ptr =
          (uint16_t
               *)&output_pad_part[y * output_partition_width * slice_depth +
                                  x * slice_depth];
      for (int c = 0; c < slice_depth; c++) {
        dest_ptr[c] =
            ((uint32_t)src_ptr[c] << 16) |
            src_ptr[c + 8]; // reconstruct from upper and lower 2 bytes
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
};

void vec_init() {
  rspq_init();
  // Register the overlay
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }

void generate_padded_slices_with_depth_slice(
    int8_t *data, int8_t *padded_input_partition, const int h_slice_num,
    const int w_slice_num, const int depth_slice_num, const int in_h,
    const int in_w, const int in_depth, const int slice_height,
    const int slice_width, const int padding, const int overlap) {

  // start_[h/w]_pad are the starting indices of the slice in the (virtual)
  // fully padded input data.  We then need to deterimine the starting indices
  // in the actual input data, which may be less than the padded input data.
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? padding : 0;
  const int slice_start_w = (start_w_pad == 0) ? padding : 0;
  const int num_h_elems = slice_height - slice_start_h;
  const int num_w_elems = slice_width - slice_start_w;

  // Calculate depth start and end based on the depth_slice
  const int depth_start =
      depth_slice_num * 8; // Start at the depth slice of 8 channels
  const int depth_end =
      MIN(depth_start + 8, in_depth); // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0,
         sizeof(int8_t) * slice_height * slice_width * 8);

  // The starting indices in the full non-padded input data
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - padding;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - padding;
  int w_end = MIN(w_start + num_w_elems, in_w);

  // Copy data to the padded slice
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        int8_t value = data[(h * in_w * in_depth) + (w * in_depth) + d];
        int slice_row = slice_start_h + (h - h_start);
        int slice_col = slice_start_w + (w - w_start);
        int slice_depth = d - depth_start;
        padded_input_partition[(slice_row * slice_width * 8) + (slice_col * 8) +
                               slice_depth] = value;
      }
    }
  }
}

static inline void
RSPDepthConvTiledPadded(int32_t *dest, int8_t *input_data, int8_t *weights,
                        int input_partition_height, int in_h, int in_w,
                        int in_c, int out_h, int out_w, int k_h, int k_w,
                        int output_partition_height, int input_partition_width,
                        int output_partition_width, int padding, int stride) {
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
  const int in_bytes = sizeof(int8_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size =
      input_partition_height * input_partition_width * 8 * in_bytes;
  int8_t *input_pad_part = malloc_uncached_aligned(8, in_part_size);

  const int out_part_size =
      output_partition_height * output_partition_width * 8;
  int32_t *output_pad_part =
      malloc_uncached_aligned(8, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  const int w_stride_slice = 8 * in_bytes * stride;
  const int w_slide_byte_offset = 8 * in_bytes * (input_partition_width);
  const int h_slide_byte_offset =
      (k_w * 8 * in_bytes) - (in_bytes * 8) +
      (stride - 1) * 8 * (in_h + 2 * padding) * in_bytes;

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width,
             w_stride_slice, w_slide_byte_offset, h_slide_byte_offset,
             w_part_size, in_part_size);

  const int num_h_partitions = ceil((float)(in_h + 2 * padding - overlap) /
                                    (input_partition_height - overlap));

  const int num_w_partitions = ceil((float)(in_w + 2 * padding - overlap) /
                                    (input_partition_width - overlap));

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int remaining_out_values =
        out_h * out_w * 8; // Remaining values to copy for this depth slice
    int max_output_partition_height = output_partition_height;

    // Copy weights for this depth slice to the RSP once
    rspq_write(vec_id, DMAWeights,
               PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    /* printf("Weight slice %d: \n", depth_slice); */
    /* printInt8ArrayHWC(&weights[depth_slice * k_h * k_w * 8], k_h, k_w, 8); */

    // Generate first input  slice
    generate_padded_slices_with_depth_slice(
        input_data, input_pad_part, 0, 0, depth_slice, in_h, in_w, in_c,
        input_partition_height, input_partition_width, padding, overlap);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions;
         h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions;
           w_slice_count++) {
        rspq_wait();

        /* printf("Input slice %d, %d : \n", h_slice_count, w_slice_count); */
        /* printInt8ArrayHWC(input_pad_part, input_partition_height, */
        /*                   input_partition_width, 8); */

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part),
                   out_part_size * out_bytes);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_padded_slices_with_depth_slice(
              input_data, input_pad_part, h_slice_count, w_slice_count + 1,
              depth_slice, in_h, in_w, in_c, input_partition_height,
              input_partition_width, padding, overlap);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_padded_slices_with_depth_slice(
              input_data, input_pad_part, h_slice_count + 1, 0, depth_slice,
              in_h, in_w, in_c, input_partition_height, input_partition_width,
              padding, overlap);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part),
                   in_part_size);
        /* printf("DMA inputs again!\n"); */

        copy_output_slice_to_full(
            dest, output_pad_part, output_partition_height,
            output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
            depth_slice, max_output_partition_height);

        remaining_out_values -=
            output_partition_height * output_partition_width * 8;

        /* printf("Output slice %d, %d: \n", h_slice_count, w_slice_count); */
        /* printInt32ArrayHWC_reorder(output_pad_part, output_partition_height,
         */
        /*                            output_partition_width, 8); */
        /* if (h_slice_count == 1 && w_slice_count == 1) { */
        /*   goto end; */
        /*   /\* printf("Output slice %d, %d: \n", h_slice_count,
         * w_slice_count); *\/ */
        /*   /\* printInt32ArrayHWC_reorder(output_pad_part, */
        /*    * output_partition_height, *\/ */
        /*   /\*                            output_partition_width, 8); *\/ */
        /* } */
      }

      // May need to adjust the max_output_partition_height for the final
      // slice(s)
      if (remaining_out_values < (out_part_size * num_w_partitions)) {
        // Cover the case where our final output slice is larger than required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
    }
  }
  free_uncached(input_pad_part);
  free_uncached(output_pad_part);
}

void depthwise_convolution(const int input_height, const int input_width,
                           const int input_depth, const int kernel_height,
                           const int kernel_width, const int stride,
                           const int padding, const int input_partition_height,
                           const int input_partition_width,
                           const int output_partition_height,
                           const int output_partition_width) {
  int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
  int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

  printf("Allocating arrays for depthwise convolution\n");
  printf("Input dimensions: %d x %d x %d\n", input_height, input_width,
         input_depth);
  printf("Padding: %d, stride: %d\n", padding, stride);
  printf("Kernel dimensions: %d x %d\n", kernel_height, kernel_width);
  printf("Output shape: %d x %d x %d\n", output_height, output_width,
         input_depth);

  // Allocate and initialize input, output and weights
  const int wbytes = sizeof(int8_t);
  const int in_bytes = sizeof(int8_t);
  const int out_bytes = sizeof(int32_t);
  int8_t *input_data = malloc_uncached_aligned(8, input_height * input_width *
                                                      input_depth * in_bytes);
  int32_t *output = malloc_uncached_aligned(8, output_height * output_width *
                                                   input_depth * out_bytes);
  int32_t *output_target = malloc_uncached_aligned(
      8, output_height * output_width * input_depth * out_bytes);
  int8_t *weights = malloc_uncached_aligned(8, kernel_height * kernel_width *
                                                   input_depth * wbytes);

  // Initialize input data and weights with sequential values for the
  // example (not there will be overflow, but int32 accumulation won't care)
  for (int i = 0; i < input_height * input_width * input_depth; i++) {
    input_data[i] = i;
  }

  for (int i = 0; i < kernel_height * kernel_width * input_depth; i++) {
    weights[i] = i;
  }

  /* printf("Weights\n"); */
  /* printInt8ArrayHWC(weights, kernel_height, kernel_width, input_depth);
   */

  int8_t *weights_reshape = malloc_uncached_aligned(
      8, kernel_height * kernel_width * input_depth * wbytes);
  offline_weight_reshape(weights, weights_reshape, kernel_height, kernel_width,
                         input_depth, 8);

  /* printf("Weights reshape\n"); */
  /* printIntArrayHWC(weights_reshape, kernel_height, kernel_width,
   * input_depth); */
  /* printf("Input\n"); */
  /* printIntArrayHWC(input_data, input_height, input_width, input_depth);
   */

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
      kernel_width, output_partition_height, input_partition_width,
      output_partition_width, padding, stride);
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

  /* printf("Output_target\n"); */
  /* printInt32ArrayHWC(output_target, output_height, output_width,
   * input_depth); */

  if (differences == 0) {
    printf("The RSP computation is correct (speedup %f).\n",
           (float)time_spent_cpu / (float)time_spent_rsp);
  } else {
    printf("There are %d differences. The RSP computation is not correct.\n",
           differences);
  }
  free_uncached(input_data);
  free_uncached(output);
  printf("---------------------------------\n\n");
}

void tests() {
  depthwise_convolution(
      /*input_height=*/4, /*input_width=*/4,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/4,
      /*input_partition_width=*/4, /*output_partition_height=*/2,
      /*output_partition_width=*/2);

  depthwise_convolution(
      /*input_height=*/4, /*input_width=*/4,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/4,
      /*input_partition_width=*/4, /*output_partition_height=*/2,
      /*output_partition_width=*/2);

  depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/6,
      /*output_partition_width=*/6);

  depthwise_convolution(
      /*input_height=*/10, /*input_width=*/10,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/7,
      /*input_partition_width=*/12, /*output_partition_height=*/5,
      /*output_partition_width=*/10);

  depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/3,
      /*input_partition_width=*/32, /*output_partition_height=*/1,
      /*output_partition_width=*/30);

  depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/3,
      /*input_partition_width=*/32, /*output_partition_height=*/1,
      /*output_partition_width=*/30);

  depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/5,
      /*input_partition_width=*/18, /*output_partition_height=*/3,
      /*output_partition_width=*/16);

  depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/5,
      /*input_partition_width=*/10, /*output_partition_height=*/3,
      /*output_partition_width=*/8);
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

  tests();

  // Clean up
  vec_close();

  return 0;
}
