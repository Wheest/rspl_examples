BUILD_DIR=build
include $(N64_INST)/include/n64.mk

all: rspsimple.z64

$(BUILD_DIR)/rspsimple.elf: $(BUILD_DIR)/main.o $(BUILD_DIR)/rsp_simple.o

rspsimple.z64: N64_ROM_TITLE="RSPQ Demo"

clean:
	rm -rf $(BUILD_DIR) rspsimple.z64

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all clean
