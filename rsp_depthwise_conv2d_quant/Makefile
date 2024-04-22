BUILD_DIR=build
include $(N64_INST)/include/n64.mk

all: rsp_depth_quant.z64

$(BUILD_DIR)/rsp_depth_quant.elf: $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/rsp_simple.o

rsp_depth_quant.z64: N64_ROM_TITLE="RSPQ Demo"

clean:
	rm -rf $(BUILD_DIR) rsp_depth_quant.z64

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all clean
