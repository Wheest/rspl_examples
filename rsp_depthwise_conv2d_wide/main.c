#include "utils.h"
#include <libdragon.h>
#include <math.h>
#include <stdio.h>

#define RSPQ_DEBUG 1
#define RSPQ_PROFILE 1
#define PRINT_DEBUG 0
#define OUTPUT_PRINT 0

#define MIN(a, b) ((a) < (b) ? (a) : (b))
DEFINE_RSP_UCODE(rsp_simple);

uint32_t vec_id;

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
    const int slice_width, const int pad_t, const int pad_l,
    const int overlap) {

  // start_[h/w]_pad are the starting indices of the slice in the (virtual)
  // fully padded input data.  We then need to deterimine the starting indices
  // in the actual input data, which may be less than the padded input data.
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? pad_t : 0;
  const int slice_start_w = (start_w_pad == 0) ? pad_l : 0;
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
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - pad_t;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - pad_l;
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
                        int output_partition_width, int pad_t, int pad_b,
                        int pad_l, int pad_r, int stride) {
  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  if (PRINT_DEBUG) {
    printf("Weights reshape\n");
    printInt8ArrayHWC(weights, 3, 3, 8);
  }

  const int wbytes = sizeof(int8_t);
  const int in_bytes = sizeof(int8_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size =
      input_partition_height * input_partition_width * 8 * in_bytes;
  int8_t *input_pad_part = malloc_uncached_aligned(16, in_part_size);

  const int out_part_size =
      output_partition_height * output_partition_width * 8;
  int32_t *output_pad_part =
      malloc_uncached_aligned(16, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;

  // the number of bytes to offset our input data pointer by between elements
  // in the same window.  I.e., if our pointer is at 0, how does it get to 5?
  // |*0, 1, 2,| 3, 4
  // | 5, 6, 7,| 8, 9
  // |10,11,12,|13,14
  //  15,16,17,18,19
  const int w_slide_byte_offset = (8 * in_bytes * input_partition_width);

  // when we slide right for a new window, the number of bytes to offset our
  // input data pointer
  const int w_window_stride = 8 * in_bytes * stride;
  // when we slide  for a new window, the number of bytes to offset our
  // input data pointer
  const int h_window_stride =
      (8 * in_bytes) *                  /*account for our depth of 8*/
      ((input_partition_width * stride) /*move the pointer vertically down*/
       - /*move the pointer back left to the start of the row*/
       (output_partition_width * stride));

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width,
             w_window_stride, w_slide_byte_offset, h_window_stride, w_part_size,
             in_part_size);

  const int num_h_partitions = ceil((float)(in_h + pad_t + pad_b - overlap) /
                                    (input_partition_height - overlap));

  const int num_w_partitions = ceil((float)(in_w + pad_l + pad_r - overlap) /
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
        input_partition_height, input_partition_width, pad_t, pad_l, overlap);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions;
         h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions;
           w_slice_count++) {
        rspq_wait();

        if (PRINT_DEBUG) {
          printf("Input slice %d, %d : \n", h_slice_count, w_slice_count);
          printInt8ArrayHWC(input_pad_part, input_partition_height,
                            input_partition_width, 8);
        }

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part),
                   out_part_size * out_bytes);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_padded_slices_with_depth_slice(
              input_data, input_pad_part, h_slice_count, w_slice_count + 1,
              depth_slice, in_h, in_w, in_c, input_partition_height,
              input_partition_width, pad_t, pad_l, overlap);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_padded_slices_with_depth_slice(
              input_data, input_pad_part, h_slice_count + 1, 0, depth_slice,
              in_h, in_w, in_c, input_partition_height, input_partition_width,
              pad_t, pad_l, overlap);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part),
                   in_part_size);

        copy_output_slice_to_full(
            dest, output_pad_part, output_partition_height,
            output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
            depth_slice, max_output_partition_height);

        remaining_out_values -=
            output_partition_height * output_partition_width * 8;

        if (PRINT_DEBUG) {
          printf("Output slice %d, %d: \n", h_slice_count, w_slice_count);
          /* debug_hexdump(output_pad_part, out_part_size * out_bytes); */
          printInt32ArrayHWC_reorder(output_pad_part, output_partition_height,
                                     output_partition_width, 8);
        }
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

bool depthwise_convolution_extended_padding(
    const int input_height, const int input_width, const int input_depth,
    const int kernel_height, const int kernel_width, const int stride,
    const int pad_t, const int pad_b, const int pad_l, const int pad_r,
    const int input_partition_height, const int input_partition_width,
    const int output_partition_height, const int output_partition_width) {
  int output_height =
      (input_height - kernel_height + pad_t + pad_b) / stride + 1;
  int output_width = (input_width - kernel_width + pad_l + pad_r) / stride + 1;

  printf("Allocating arrays for depthwise convolution\n");
  printf("Input dimensions: %d x %d x %d\n", input_height, input_width,
         input_depth);
  printf("Padding: (%d, %d, %d, %d), stride: %d\n", pad_t, pad_l, pad_b, pad_r,
         stride);
  printf("Kernel dimensions: %d x %d\n", kernel_height, kernel_width);
  printf("Output shape: %d x %d x %d\n", output_height, output_width,
         input_depth);

  // assert that the partitioning is less than or equal to the input and
  // output
  if (input_partition_height > (input_height + pad_t + pad_b)) {
    printf("Error: input_partition_height %d must be less than or equal to "
           "input_height (%d) + pad_t + pad_b\n",
           input_partition_height, input_height);
    return false;
  }
  if (input_partition_width > (input_width + pad_l + pad_r)) {
    printf("Error: input_partition_width %d must be less than or equal to "
           "input_width (%d) + pad_l + pad_r\n",
           input_partition_width, input_width);
    return false;
  }
  if (output_partition_height > output_height) {
    printf("Error: output_partition_height %d must be less than or equal to "
           "output_height %d\n",
           output_partition_height, output_height);
    return false;
  }
  if (output_partition_width > output_width) {
    printf("Error: output_partition_width %d must be less than or equal to "
           "output_width %d\n",
           output_partition_width, output_width);
    return false;
  }

  // Allocate and initialize input, output and weights
  const int wbytes = sizeof(int8_t);
  const int in_bytes = sizeof(int8_t);
  const int out_bytes = sizeof(int32_t);
  int8_t *input_data = malloc_uncached_aligned(16, input_height * input_width *
                                                       input_depth * in_bytes);
  int32_t *output = malloc_uncached_aligned(16, output_height * output_width *
                                                    input_depth * out_bytes);
  int32_t *output_target = malloc_uncached_aligned(
      8, output_height * output_width * input_depth * out_bytes);
  int8_t *weights = malloc_uncached_aligned(16, kernel_height * kernel_width *
                                                    input_depth * wbytes);

  // Initialize input data and weights with sequential values for the
  // example (not there will be overflow, but int32 accumulation won't care)
  for (int i = 0; i < input_height * input_width * input_depth; i++) {
    input_data[i] = i;
  }

  if (PRINT_DEBUG) {
    printf("input_data\n");
    printInt8ArrayHWC(input_data, input_height, input_width, input_depth);
  }

  for (int i = 0; i < kernel_height * kernel_width * input_depth; i++) {
    weights[i] = i;
  }

  /* printf("Weights\n"); */
  /* printInt8ArrayHWC(weights, kernel_height, kernel_width, input_depth);
   */

  int8_t *weights_reshape = malloc_uncached_aligned(
      16, kernel_height * kernel_width * input_depth * wbytes);
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
                                   output_width, stride, pad_t, pad_l);
  unsigned long time_spent_cpu = get_ticks() - start;
  printf("\nCPU baseline time (CPU ticks): %lu\n", time_spent_cpu);

  rspq_wait();
  start = get_ticks();
  RSPDepthConvTiledPadded(
      output, input_data, weights_reshape, input_partition_height, input_height,
      input_width, input_depth, output_height, output_width, kernel_height,
      kernel_width, output_partition_height, input_partition_width,
      output_partition_width, pad_t, pad_b, pad_l, pad_r, stride);
  rspq_wait();
  unsigned long time_spent_rsp = get_ticks() - start;
  printf("\nRSP time (CPU ticks): %lu\n", time_spent_rsp);

  // Compare the results
  int differences = 0;
  for (int i = 0; i < output_height * output_width * input_depth; ++i) {
    if (output[i] != output_target[i]) {
      differences++;
    }
  }

  if (PRINT_DEBUG || OUTPUT_PRINT) {
    printf("Output\n");
    printInt32ArrayHWC(output, output_height, output_width, input_depth);

    printf("Output_target\n");
    printInt32ArrayHWC(output_target, output_height, output_width, input_depth);
  }

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
  return differences == 0;
}

bool depthwise_convolution(const int input_height, const int input_width,
                           const int input_depth, const int kernel_height,
                           const int kernel_width, const int stride,
                           const int padding, const int input_partition_height,
                           const int input_partition_width,
                           const int output_partition_height,
                           const int output_partition_width) {
  // this is the case where our padding is symmetric
  const int pad_t = padding;
  const int pad_b = padding;
  const int pad_l = padding;
  const int pad_r = padding;
  return depthwise_convolution_extended_padding(
      input_height, input_width, input_depth, kernel_height, kernel_width,
      stride, pad_t, pad_b, pad_l, pad_r, input_partition_height,
      input_partition_width, output_partition_height, output_partition_width);
}

void tests() {
  bool all_passed = true;
  all_passed &= depthwise_convolution(
      /*input_height=*/4, /*input_width=*/4,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/4,
      /*input_partition_width=*/4, /*output_partition_height=*/2,
      /*output_partition_width=*/2);

  all_passed &= depthwise_convolution(
      /*input_height=*/4, /*input_width=*/4,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/4,
      /*input_partition_width=*/4, /*output_partition_height=*/2,
      /*output_partition_width=*/2);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/6,
      /*output_partition_width=*/6);

  all_passed &= depthwise_convolution(
      /*input_height=*/10, /*input_width=*/10,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/7,
      /*input_partition_width=*/12, /*output_partition_height=*/5,
      /*output_partition_width=*/10);

  all_passed &= depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/3,
      /*input_partition_width=*/32, /*output_partition_height=*/1,
      /*output_partition_width=*/30);

  all_passed &= depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/3,
      /*input_partition_width=*/32, /*output_partition_height=*/1,
      /*output_partition_width=*/30);

  all_passed &= depthwise_convolution(
      /*input_height=*/32,
      /*input_width=*/32,
      /*input_depth=*/16, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/5,
      /*input_partition_width=*/18, /*output_partition_height=*/3,
      /*output_partition_width=*/16);

  all_passed &= depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/5,
      /*input_partition_width=*/18, /*output_partition_height=*/3,
      /*output_partition_width=*/16);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/7,
      /*input_partition_width=*/10, /*output_partition_height=*/5,
      /*output_partition_width=*/8);

  all_passed &= depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/6,
      /*input_partition_width=*/16, /*output_partition_height=*/4,
      /*output_partition_width=*/14);

  all_passed &= depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/5,
      /*input_partition_width=*/16, /*output_partition_height=*/3,
      /*output_partition_width=*/14);

  all_passed &= depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/5,
      /*input_partition_width=*/16, /*output_partition_height=*/3,
      /*output_partition_width=*/14);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/192, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/9,
      /*input_partition_width=*/10, /*output_partition_height=*/7,
      /*output_partition_width=*/8);

  all_passed &= depthwise_convolution(
      /*input_height=*/8,
      /*input_width=*/8,
      /*input_depth=*/192,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*stride=*/1,
      /*padding=*/1,
      /*input_partition_height=*/9,
      /*input_partition_width=*/10,
      /*output_partition_height=*/7,
      /*output_partition_width=*/8);

  all_passed &= depthwise_convolution(
      /*input_height=*/16, /*input_width=*/16,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/5,
      /*input_partition_width=*/16, /*output_partition_height=*/3,
      /*output_partition_width=*/14);

  all_passed &= depthwise_convolution(
      /*input_height=*/32, /*input_width=*/32,
      /*input_depth=*/32, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/1, /*input_partition_height=*/5,
      /*input_partition_width=*/18, /*output_partition_height=*/3,
      /*output_partition_width=*/16);

  // Stride 2 test cases
  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0,
      /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/3,
      /*output_partition_width=*/3);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0, /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/3,
      /*output_partition_width=*/3);

  all_passed &= depthwise_convolution(
      /*input_height=*/4, /*input_width=*/4,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0, /*input_partition_height=*/4,
      /*input_partition_width=*/4, /*output_partition_height=*/1,
      /*output_partition_width=*/1);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0, /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/3,
      /*output_partition_width=*/3);

  all_passed &= depthwise_convolution(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*padding=*/0, /*input_partition_height=*/8,
      /*input_partition_width=*/8, /*output_partition_height=*/6,
      /*output_partition_width=*/6);

  all_passed &= depthwise_convolution(
      /*input_height=*/9, /*input_width=*/9,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0, /*input_partition_height=*/9,
      /*input_partition_width=*/9, /*output_partition_height=*/4,
      /*output_partition_width=*/4);

  all_passed &= depthwise_convolution(
      /*input_height=*/5, /*input_width=*/5,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/2, /*padding=*/0, /*input_partition_height=*/5,
      /*input_partition_width=*/5, /*output_partition_height=*/2,
      /*output_partition_width=*/2);

  // Assymetric padding tests
  all_passed &= depthwise_convolution_extended_padding(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*pad_t=*/1, /*pad_b=*/0, /*pad_l=*/1,
      /*pad_r=*/0, /*input_partition_height=*/9,
      /*input_partition_width=*/9, /*output_partition_height=*/7,
      /*output_partition_width=*/7);

  all_passed &= depthwise_convolution_extended_padding(
      /*input_height=*/8, /*input_width=*/8,
      /*input_depth=*/8, /*kernel_height=*/3, /*kernel_width=*/3,
      /*stride=*/1, /*pad_t=*/0, /*pad_b=*/1, /*pad_l=*/0,
      /*pad_r=*/1, /*input_partition_height=*/9,
      /*input_partition_width=*/9, /*output_partition_height=*/7,
      /*output_partition_width=*/7);

  // print if all passed
  if (all_passed) {
    printf("All tests passed\n");
  } else {
    printf("Some tests failed\n");
  }
}
int main() {
  // Initialize systems
  console_init();
  console_set_debug(true);
  debug_init_isviewer();
  debug_init_usblog();

  vec_init();
  printf("Init'd RSP overlay\n");

  tests();

  // Clean up
  vec_close();

  return 0;
}
