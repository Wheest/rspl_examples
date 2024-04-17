all: async_rom_load.z64
.PHONY: all

BUILD_DIR = build
include $(N64_INST)/include/n64.mk

OBJS = $(BUILD_DIR)/async_rom_load.o

async_rom_load.z64: N64_ROM_TITLE = "async_rom_load"
async_rom_load.z64: $(BUILD_DIR)/async_rom_load.dfs

$(BUILD_DIR)/async_rom_load.dfs: $(wildcard filesystem/*)
$(BUILD_DIR)/async_rom_load.elf: $(OBJS)

clean:
	rm -rf $(BUILD_DIR) *.z64
.PHONY: clean

-include $(wildcard $(BUILD_DIR)/*.d)
