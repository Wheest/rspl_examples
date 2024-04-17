#!/usr/bin/env python3

import struct
import os

# create the filesystem directory if it does not exist
os.makedirs("filesystem", exist_ok=True)

# Open a file in binary write mode
with open("filesystem/integers.dat", "wb") as file:
    # Iterate over the range of 1024 8-bit integers
    for i in range(1024):
        # Ensure the integer is within the 8-bit range
        value = i % 256
        # Pack the integer as an unsigned char ('B') and write to the file
        file.write(struct.pack("B", value))
