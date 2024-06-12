#!/usr/bin/env python3

import math


def get_partition_size(in_shape, padding, stride):
    # in_shape = (batch, in_h, in_w, in_c)
    in_w = in_shape[2]
    in_h = in_shape[1]
    assert in_w == in_h
    in_dbytes = 2
    out_dbytes = 4
    kdim_h = 3
    kdim_w = 3
    out_w = math.floor((in_w + 2 * padding - kdim_w) / stride) + 1
    out_h = math.floor((in_h + 2 * padding - kdim_h) / stride) + 1
    print("out_h:", out_h)
    print("out_w:", out_w)

    kmem = kdim_h * kdim_w * 8 * in_dbytes
    overlap = kdim_w - stride
    input_part_width_base = in_w + (2 * padding)
    overhead = 500
    max_mem = (4 * 1024) - overhead

    input_part_height = 3
    input_part_width = input_part_width_base
    output_part_width = out_w
    input_mem = input_part_height * input_part_width * 8 * in_dbytes
    output_part_height = (input_part_height - kdim_h + 2 * padding) // stride + 1
    omem = output_part_height * output_part_width * 8 * out_dbytes

    spare_room = max_mem - (kmem + omem)

    if input_mem > spare_room:
        split_factor = 1
        while True:
            split_factor += 1
            input_part_width = math.floor(
                (input_part_width_base + overlap) / split_factor
            )
            output_part_width = (input_part_width - kdim_w) // stride + 1
            input_mem = input_part_height * input_part_width * 8 * in_dbytes
            output_part_height = (
                input_part_height - kdim_h + 2 * padding
            ) // stride + 1
            omem = output_part_height * output_part_width * 8 * out_dbytes
            spare_room = max_mem - (kmem + omem)
            if input_mem <= spare_room:
                break

    loop_count = 0
    while (kmem + input_mem + omem) <= max_mem and input_part_height < (
        in_h + 2 * padding
    ):
        input_part_height += stride
        if input_part_height > (in_h + 2 * padding):
            input_part_height = in_h + 2 * padding
        input_mem = input_part_height * input_part_width * 8 * in_dbytes
        output_part_height = math.floor(
            (min((in_h + 2 * padding), input_part_height) - kdim_h) / stride + 1
        )
        print("1: output_part_height:", output_part_height)
        omem = output_part_height * output_part_width * 8 * out_dbytes
        loop_count += 1
        if loop_count > 1000:
            raise RuntimeError("Loop count exceeded 1000")

    loop_count = 0
    while (kmem + input_mem + omem) > max_mem:
        input_part_height -= stride
        if input_part_height > (in_h + 2 * padding):
            input_part_height = in_h + 2 * padding
        input_mem = input_part_height * input_part_width * 8 * in_dbytes
        print("1: output_part_height:", output_part_height)
        output_part_height = math.floor(
            (min((in_h + 2 * padding), input_part_height) - kdim_h) / stride + 1
        )
        omem = output_part_height * output_part_width * 8 * out_dbytes
        if loop_count > 1000:
            raise RuntimeError("Loop count exceeded 1000")

    print(
        f"at this point:available memory {kmem}, {input_mem}, {omem}, {max_mem}, {kmem + input_mem + omem}"
    )
    num_h_partitions = math.ceil(
        (in_h + 2 * padding - overlap) / (input_part_height - overlap)
    )
    num_h_partitions_2 = math.ceil(
        (in_h + 2 * padding - overlap) / max(1, (input_part_height - 1 - overlap))
    )
    print(num_h_partitions, num_h_partitions_2)

    if num_h_partitions == num_h_partitions_2:
        print("1:", input_part_height)
        input_part_height = (
            math.floor((in_h + 2 * padding) / num_h_partitions) + overlap
        )
        print(
            f"{in_h} + 2 * {padding} / {num_h_partitions} + {overlap} = {input_part_height}"
        )
        print("2", input_part_height)
        input_part_height = min(input_part_height, in_h + 2 * padding)
        print("3", input_part_height)

        input_mem = input_part_height * input_part_width * 8 * in_dbytes
        print("1: output_part_height:", output_part_height)
        output_part_height = math.floor(
            (min((in_h + 2 * padding), input_part_height) - kdim_h) / stride + 1
        )
        print("2: output_part_height:", output_part_height)
        omem = output_part_height * output_part_width * 8 * out_dbytes

    if (kmem + input_mem + omem) > max_mem:
        raise MemoryError(
            f"Memory usage exceeds maximum available memory {kmem}, {input_mem}, {omem}, {max_mem}, {kmem + input_mem + omem}"
        )
    elif input_part_height < 3:
        raise ValueError("Input partition height is less than 3")

    if input_part_width < 3:
        raise ValueError("Input partition width is less than 3")
    if input_part_height > (in_h + 2 * padding):
        raise ValueError("Input partition height exceeds input height")
    if input_part_width > (in_w + 2 * padding):
        raise ValueError("Input partition width exceeds input width")
    if output_part_height > out_w:
        raise ValueError("Output partition height exceeds output height")
    if output_part_width > out_w:
        raise ValueError("Output partition width exceeds output width")

    print("input part height:", input_part_height)
    print("input part width:", input_part_width)
    print("output part height:", output_part_height)
    print("output part width:", output_part_width)
    return input_part_height, input_part_width, output_part_height, output_part_width


# in_shape = (1, 32, 32, 16)
# padding = 1
# stride = 1
# get_partition_size(in_shape, padding, nstride)


# in_shape = (1, 4, 4, 8)
# padding = 0
# stride = 2
# get_partition_size(in_shape, padding, stride)


in_shape = (1, 8, 8, 8)
padding = 0
stride = 2
get_partition_size(in_shape, padding, stride)


in_shape = (1, 8, 8, 8)
padding = 0
stride = 1
get_partition_size(in_shape, padding, stride)


#
in_shape = (1, 32, 32, 144)
padding = 0
stride = 2
get_partition_size(in_shape, padding, stride)


# in_shape = (1, 8, 8, 8)
# padding = 1
# stride = 1
# get_partition_size(in_shape, padding, stride)
