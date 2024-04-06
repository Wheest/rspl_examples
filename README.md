<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Wheest/rspl_examples">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">RSPL examples</h3>

  <p align="center">
    A collection of examples of using the RSPL language for the N64
    <br />
  </p>
</div>

The following are some functional examples of the RSPL language for the N64's RSP co-processor.
A web app with docs and transpiler for RSPL is [available here](https://mbeboek.gitlab.io/rspl/), which explains more about it.
Big props to Max Bebök (HailToDodongo) for all their work on it.
The language is great, because the alternative for writing accelerated RSP code on the N64 is raw MIPS assembly, which is time consuming to write.

I made this repository because there were not a lot of small self-contained examples for the RSPL language, and I was developing accelerated kernels for my [DNN64 project](https://gibsonic.org/blog/2024/03/12/dnn64_p1).

## Add One

The simplest example, which shows how we can use DMA to copy in an array of values and increment all of them by one.

## Reduction

Shows how we can do a vector reduction (i.e., the sum of a vector register).
There isn't a native instruction for this, so we need to use a trick in three instructions:

``` javascript
vec += vec.yywwYYWW;
vec += vec.zzzzZZZZ;
vec += vec.X;
```

Checkout [my Twitter thread](https://twitter.com/PerryGibson_/status/1776239662136185044) if you want to understand why this works.

## int32 accumulation

When we do lots of multiplications (and accumulations), there's a risk that we can overflow (e.g., multiply two 16 bit values, and you could require 32 bits).
On the RSP, we can handle this, which is what this example shows.

However, a word of warning.
This elementwise multiplication of two int16 vector registers (with 8 values in each) stores the result in two vector registers, since we need 256 bits for 8 32-bit values.
But the format is not what we need, it stores the upper 2 bytes of each element first, and then the lower 2 bytes.

This means that to get our values, we need move the bytes around.
I chose to do this on the CPU-side (using the `reconstruct_vector` function), as it makes more sense for my usecase.
However, there are ways of doing it on the RSP-side.  Checkout the [N64brew Discord](https://n64brew.dev/wiki/Main_Page) for more information.


## int16 8x8 matmul

This example does matmul (`C = A * B`) with two int16 8x8 matrices.
It assumes that overflow is not a problem (otherwise you would need to add int32 accumulation like the above example).
We calculate in row-major, with the B matrix transposed to take advantage of the vector instructions.

## Depthwise convolution

The following are examples of running the [depthwise convolution operation](https://paperswithcode.com/method/depthwise-convolution), at increasing levels of complexity.
E.g., the first version assumes a fixed data size, that can fit on the RSP completely; whereas v2 allows larger datasizes, but fixed strides, etc.

### Depthwise conv2d (v1)

Implements channels last depthwise convolution, for int16, accumulating onto int16 (so assumes no overflow).
The inputs are of shape `(4, 4, 8)`, the kernel size is `3x3`, the strides are 1, and the padding is 0.

### Depthwise conv2d large (v2)

This example adds support for depthwise convolutions that are too large along the height and width dimensions.
A Python Jinja script (`conv2d_rspl_gen.py`) is used to create the RSPL code for your given input size.
Unlike v1, v2 works with padding, but only stride 1, and depth 8.
It runs computations in a partitioned fashion, as you can see in this diagram which divides the input data into 6 overlapping partitions:

<img src="./rsp_depthwise_conv2d_large/partition_6.png" alt="Example of partitions" width="30%" />

Partitioning is necessary because the N64 has a limited amount of memory, and partition 6 is the same size as the other 5, but with extra zeros at the end which are discarded in the final output.
The maximum height of the input and output partitions are determined by `conv2d_rspl_gen.py`, which iteratively guestimates how much memory the RSP would have left.

The script (`conv2d_rspl_gen.py`) prints what the suggested partition sizes are, and you should edit them in the `main.c` file as appropriate.
You need to manually copy the generated RSPL file to the [RSPL webapp](https://mbeboek.gitlab.io/rspl/), then copy the generated assembly into the `rsp_simple.S` file.
It is not until version 3 of the script (adding support for deeper depthwise convolutions) that I add automatic compilation.

### Depthwise conv2d deep (v3)

This example adds support for deeper depthwise convolution, i.e., `in_c >= 8`.
It still needs to be a multiple of 8, but in my experience it usually is anyway.
It also only supports stride of 1.

It computes partitions of depth 8 in the same way as v2, but also computes other partitions of depth 8.
It computes all the outputs at a given depth first, then loads the next set of weights, and computes the next set.

An example of this is given below, where we compute the red, green, and blue partitions first, before moving back to the next set of 3 partitions.

![Example of depth partitions](./rsp_depthwise_conv_deep/deep_partitions.png)

The RSPL code generation script (`conv2d_rspl_gen.py`) has been upgraded (I'm not backporting lol), and now calls the RSPL compiler itself, rather than generating an RSPL file that you need to [copy into the web app manually](https://mbeboek.gitlab.io/rspl/).
See [the RSPL GitLab](https://gitlab.com/mbeboek/rspl) for instructions for building, and be sure to replace the path to the CLI in the code generation script.

### Depthwise conv2d stride (v4)

This example extends v3 to support strides for depthwise convolution.
When using a stride greater than 1, the average speedups go from ~8.4x to ~4.4x, since there is now less data reuse.

The RSPL code generation script (`conv2d_rspl_gen.py`) has been upgraded, and now also better calculates the state memory available and being used by our kernel, so that it suggests larger partition sizes.
