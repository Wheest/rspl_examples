#!/usr/bin/env python3
import numpy as np
import math
import jinja2
import subprocess


def generate_rspl_file(template_vars):
    req_vars = [
        "in_size",
        "w_size",
        "osize",
        "in_h",
        "in_w",
        "oh",
        "ow",
        "padding",
        "max_h_iters",
        "kdim_w",
        "w_slide_byte_offset",
        "h_slice_byte_offset",
    ]
    for var in req_vars:
        assert var in template_vars, f"Missing required variable {var}"

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template("conv2d_dev.rspl")

    # Render the template with the given variables
    outputText = template.render(template_vars)

    # Write the render output to new file
    with open("rsp_simple.rspl", "w") as f:
        f.write(outputText)

    print("Generated template saved as conv2d_custom.rspl")


def calculate_partitions(in_h, padding, kdim_h, stride, max_height):
    # Calculate the total height including padding
    total_padded_height = in_h + 2 * padding

    # The number of rows that will be overlapped due to the kernel - stride + 1
    overlap = max(kdim_h - stride, 0)

    # Calculate the number of partitions
    # Start with the first partition
    partitions = 1
    current_end = max_height  # End row of the current partition

    while current_end < total_padded_height:
        # Slide the window by stride until we need a new partition
        start = (
            current_end - overlap
        )  # We can reuse the overlap from the previous partition
        current_end = start + max_height

        # Every loop corresponds to one additional partition needed
        partitions += 1

    return partitions


def generate_large_depth_conv2d(inputs, weights, stride, padding, force_partition=None):
    """Generate a custom RSPL template for a depthwise convolution with large memory usage
    i.e., we cannot fit everything in the RSP memory
    """
    dbytes = 2
    in_h, in_w, in_c = inputs.shape
    kdim_h, kdim_w, in_cc = weights.shape
    out_h = (in_h - kdim_h + 2 * padding) // stride + 1
    out_w = (in_w - kdim_w + 2 * padding) // stride + 1
    kmem = kdim_h * kdim_w * 8 * dbytes
    xmem = (in_h + 2 * padding) * (in_w + 2 * padding) * 8 * dbytes
    omem = out_h * out_w * 8 * dbytes

    # Split the input into chunks, so that we process windows across the width first,
    # and as much of the height as possible. This is to minimize the memory usage.
    # We then need to do additional calls to process the rest of the height.
    # This is a simple heuristic, and can be improved.
    # We can also save memory by reducing the space used for the output
    # which is dependent on the amount on input data we have
    # thus they can be calculated dynamically
    max_height = 3
    curr_mem = max_height * in_w * 8 * dbytes
    new_oh = (max_height - kdim_h + 2 * padding) // stride + 1
    omem_new = new_oh * out_w * 8 * dbytes
    # max_mem = (4 * 1024) - 570  # 570 is overhead
    max_mem = (4 * 1024) - 460  # 466 is overhead
    spare_room = max_mem - (kmem + omem_new)
    assert (
        curr_mem < spare_room
    ), "Cannot fit even one horizontal strip of input in the RSP memory, good luck with your special case implementation"
    while (kmem + curr_mem + omem_new) <= max_mem and max_height < (in_h + 2 * padding):
        max_height += 1
        curr_mem = max_height * (in_w + 2 * padding) * 8 * dbytes
        new_oh = (min((in_h + 2 * padding), max_height) - kdim_h) // stride + 1
        # new_oh = (max_height - kdim_h + 2 * padding) // stride + 1
        omem_new = new_oh * out_w * 8 * dbytes
        spare_room = max_mem - (kmem + omem_new)
        print("Condition?", (kmem + curr_mem + omem_new) <= max_mem)

    # roll back to the last valid height
    if (kmem + curr_mem + omem_new) > max_mem:
        max_height -= 1
        curr_mem = max_height * (in_w + 2 * padding) * 8 * dbytes
        # new_oh = (max_height - kdim_h + 2 * padding) // stride + 1
        new_oh = (min((in_h + 2 * padding), max_height) - kdim_h) // stride + 1
        omem_new = new_oh * out_w * 8 * dbytes

    # new_oh * out_w * 8 * dbytes
    print(f"new_oh: {new_oh}, out_w: {out_w}")
    print(f"Splitting input into chunks of {max_height} rows (using {curr_mem} bytes)")
    print(f"Kernel mem: {kmem}, Output mem: {omem_new}")
    print(f"Original out_h: {out_h}, parition out_h: {new_oh}")
    print(
        "Total memory usage: ",
        kmem + curr_mem + omem_new,
        "bytes",
        f"out of {max_mem} bytes",
        f"kmem: {kmem}, xmem: {curr_mem}, omem: {omem_new}",
    )
    if force_partition is not None:
        max_height = force_partition

    h_partitions = calculate_partitions(in_h, padding, kdim_h, stride, max_height)
    print(f"Hey, h_partitions is {h_partitions}")

    print(
        f"We need {h_partitions} h_partitions, output_partition_height: {new_oh}, input_partition_height {max_height}"
    )

    template_vars = {
        "in_size": (max_height * (in_w + 2 * padding)),
        "w_size": kdim_h * kdim_w,
        "osize": new_oh * out_w,
        "in_h": in_h,
        "in_w": in_w,
        "in_c": in_c,
        "in_w_pad": in_w + 2 * padding,
        "padding": padding,
        "oh": new_oh,
        "ow": out_w,
        "max_h_iters": new_oh,
        "input_slice_h": max_height,
        "kdim_w": kdim_w,
        "w_slide_byte_offset": 8 * dbytes * (in_w + 2 * padding),
        "h_slice_byte_offset": (kdim_w * 8 * dbytes)
        - 16
        + (stride - 1) * 8 * (in_h + 2 * padding) * dbytes,
        "w_stride_slice": 8 * dbytes * stride,
        "stride": stride,
    }
    generate_rspl_file(template_vars)


def generate_depth_conv2d(inputs, weights, stride, padding, force_partition=None):
    in_h, in_w, in_c = inputs.shape
    kdim_h, kdim_w, in_cc = weights.shape
    assert kdim_h == kdim_w == 3, "Only support kdim 3 for now"
    assert in_c == in_cc, "Weights are not in the right format"
    # assert stride == 1, "Not supported larger stride yet"
    # assert padding == 0, "Not supported padding yet"

    out_h = (in_h - kdim_h + 2 * padding) // stride + 1
    out_w = (in_w - kdim_w + 2 * padding) // stride + 1

    kmem = kdim_h * kdim_w * 8 * 2
    xmem = in_h * in_w * 8 * 2
    omem = out_h * out_w * 8 * 2
    max_mem = (4 * 1024) - 570  # 570 is overhead

    print(f"Warning: Memory usage exceeds 4KB of the RSP (inputs: {xmem})")
    generate_large_depth_conv2d(inputs, weights, stride, padding, force_partition)


input_shape = (32, 32, 16)
kernel_shape = (3, 3, 16)
input_data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
weights = np.arange(np.prod(kernel_shape), dtype=np.float32).reshape(kernel_shape)
generate_depth_conv2d(input_data, weights, 1, 1)


# compile file
def compile_rspl():
    command = "node"

    path_to_script = "/home/pez/proj/n64-dnn/rspl/dist/cli.mjs"
    argument_1 = "rsp_simple.rspl"
    argument_2 = "--reorder"
    argument_3 = "--optimize"

    # Full command construction
    cmd = [command, path_to_script, argument_1, argument_2, argument_3]

    # Run the command
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Check result
    if result.returncode == 0:
        print("Command executed successfully.")
        print("Output:", result.stdout)
    else:
        print("Error:", result.stderr)


compile_rspl()
print("Compiled rspl file")
