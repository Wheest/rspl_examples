#include "utils.h"

void sequential_depthwise_conv2d_simd(const int8_t *input_data, int32_t *output,
                                      const int8_t *weights, int input_height,
                                      int input_width, int input_depth,
                                      int kernel_height, int kernel_width,
                                      int output_height, int output_width,
                                      int stride, int pad_t, int pad_l) {
  int vector_size = 8; // Simulate a SIMD vector size of 8
  for (int h = 0; h < output_height; h++) {
    for (int w = 0; w < output_width; w++) {
      for (int c = 0; c < input_depth; c += vector_size) {
        int32_t vector_sum[8] = {0}; // Temporary sum for each vector

        // Loop over the kernel's height and width
        for (int i = 0; i < kernel_height; i++) {
          for (int j = 0; j < kernel_width; j++) {
            for (int v = 0; v < vector_size;
                 v++) { // This loop simulates the SIMD operation
              int input_h = h * stride + i - pad_t;
              int input_w = w * stride + j - pad_l;
              if (input_h >= 0 && input_w >= 0 && input_h < input_height &&
                  input_w < input_width) {
                vector_sum[v] +=
                    (int32_t)input_data[(input_h * input_width + input_w) *
                                            input_depth +
                                        c + v] *
                    (int32_t)
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

void printInt8ArrayHWC(int8_t *array, int height, int width, int in_c) {
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
        int8_t val = array[(h * width * in_c) + (w * in_c) + c];
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

void printInt32ArrayHWC(int32_t *array, int height, int width, int in_c) {
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
        int32_t val = array[(h * width * in_c) + (w * in_c) + c];
        if (c > 0) {
          printf(", ");
        }
        printf("%ld", val);
      }
      printf("]");
    }
    printf("]");
  }
  printf("]\n");
}

void printInt32ArrayHWC_reorder(int32_t *array, int height, int width,
                                int in_c) {
  if (in_c != 8) {
    printf("Error: in_c must be 8 for printInt32ArrayHWC_reorder\n");
    return;
  }

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
      uint16_t *arr = (uint16_t *)(array + (h * width * in_c) + (w * in_c));
      for (int c = 0; c < in_c; ++c) {
        // Retrieve the value at [h, w, c]
        // int32_t val = array[(h * width * in_c) + (w * in_c) + c];
        int32_t val = ((uint32_t)arr[c] << 16) | arr[c + 8];
        if (c > 0) {
          printf(", ");
        }
        printf("%ld", val);
      }
      printf("]");
    }
    printf("]");
  }
  printf("]\n");
}
