def get_partition_size(in_shape, padding=1, stride=1):
    in_w, in_h = in_shape[2], in_shape[1]
    kdim_h, kdim_w = 3, 3

    out_w = (in_w + 2 * padding - kdim_w) // stride + 1

    in_dbytes = 2

    out_dbytes = 4

    kmem = kdim_h * kdim_w * 8 * in_dbytes

    overlap = kdim_w - stride

    input_part_width_base = in_w + (2 * padding)

    overhead = 544  # Overhead of other data structures in memory

    max_mem = (4 * 1024) - overhead  # Total available memory - overhead

    input_part_height = 3  # start at 3 (kh) and increase

    input_part_width = input_part_width_base  # Default to the full width

    output_part_width = out_w

    curr_mem = input_part_height * input_part_width * 8 * in_dbytes

    output_part_height = (input_part_height - kdim_h + 2 * padding) // stride + 1

    omem_new = output_part_height * output_part_width * 8 * out_dbytes

    spare_room = max_mem - (kmem + omem_new)

    if curr_mem > spare_room:
        split_factor = 1
        # If we can't fit even one horizontal strip of input in memory
        # then we need to reduce the width of the input partition
        while curr_mem > spare_room:
            split_factor += 1

            input_part_width = (input_part_width_base + overlap) // split_factor
            output_part_width = (input_part_width - kdim_w) // stride + 1
            curr_mem = input_part_height * input_part_width * 8 * in_dbytes

            output_part_height = (
                input_part_height - kdim_h + 2 * padding
            ) // stride + 1
            omem_new = output_part_height * output_part_width * 8 * out_dbytes
            spare_room = max_mem - (kmem + omem_new)

    while (kmem + curr_mem + omem_new) <= max_mem and input_part_height < (
        in_h + 2 * padding
    ):
        input_part_height += 1

        curr_mem = input_part_height * input_part_width * 8 * in_dbytes

        output_part_height = (
            min((in_h + 2 * padding), input_part_height) - kdim_h
        ) // stride + 1

        omem_new = output_part_height * output_part_width * 8 * out_dbytes

        spare_room = max_mem - (kmem + omem_new)

    if (kmem + curr_mem + omem_new) > max_mem:
        input_part_height -= 1

        curr_mem = input_part_height * input_part_width * 8 * in_dbytes

        output_part_height = (
            min((in_h + 2 * padding), input_part_height) - kdim_h
        ) // stride + 1

        omem_new = output_part_height * output_part_width * 8 * out_dbytes

    print("inshape:", in_shape)
    print("input height:", input_part_height)
    print("input width:", input_part_width)

    print("output_height:", output_part_height)
    print("output_width:", output_part_width)

    print("input memory:", curr_mem)

    print("output memory:", omem_new)
    print()

    return input_part_height, input_part_width, output_part_height, output_part_width


# get_partition_size((1, 4, 4, 16), padding=0, stride=1)

# get_partition_size((1, 8, 8, 8), padding=0, stride=1)

# get_partition_size((1, 8, 8, 8), padding=1, stride=1)

# get_partition_size((1, 10, 10, 8), padding=1, stride=1)

# get_partition_size((1, 32, 32, 8), padding=0, stride=1)

# get_partition_size((1, 32, 32, 16), padding=0, stride=1)

get_partition_size((1, 32, 32, 16), padding=1, stride=1)
