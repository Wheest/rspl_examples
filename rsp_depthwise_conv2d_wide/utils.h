#ifndef UTILS_H_
#define UTILS_H_

#include <stdint.h>
#include <stdio.h>

void sequential_depthwise_conv2d_simd(const int8_t *input_data, int32_t *output,
                                      const int8_t *weights, int input_height,
                                      int input_width, int input_depth,
                                      int kernel_height, int kernel_width,
                                      int output_height, int output_width,
                                      int stride, int padding);

void printIntArrayHWC(int16_t *array, int height, int width, int in_c);
void printInt8ArrayHWC(int8_t *array, int height, int width, int in_c);
void printInt32ArrayHWC(int32_t *array, int height, int width, int in_c);
void printInt32ArrayHWC_reorder(int32_t *array, int height, int width,
                                int in_c);

#endif // UTILS_H_
