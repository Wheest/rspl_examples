#ifndef UTILS_H_
#define UTILS_H_

#include <stdint.h>
#include <stdio.h>

void sequential_depthwise_conv2d_simd(const int8_t *input_data, int32_t *output,
                                      const int8_t *weights, int input_height,
                                      int input_width, int input_depth,
                                      int kernel_height, int kernel_width,
                                      int output_height, int output_width,
                                      int stride, int pad_t, int pad_l);

void offline_weight_reshape(const int8_t *kernels, int8_t *kernel_reshape,
                            int kdim_h, int kdim_w, int out_c,
                            int split_factor);

void copy_output_slice_to_full(int32_t *dest, const int32_t *output_pad_part,
                               const int output_partition_height,
                               const int output_partition_width,
                               const int out_w, const int h_slice_num,
                               const int w_slice_num, const int in_depth,
                               const int depth_slice_num,
                               const int max_output_partition_height);

void printIntArrayHWC(int16_t *array, int height, int width, int in_c);
void printInt8ArrayHWC(int8_t *array, int height, int width, int in_c);
void printInt32ArrayHWC(int32_t *array, int height, int width, int in_c);
void printInt32ArrayHWC_reorder(int32_t *array, int height, int width,
                                int in_c);

#endif // UTILS_H_
