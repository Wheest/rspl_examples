#include "utils.h"

void sequential_depthwise_conv2d_simd(const int16_t *input_data,
                                      int16_t *output, const int8_t *weights,
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
