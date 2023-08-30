#
# this file was created by a computer. trust it.
#

# compiler tools
XILINX_VITIS ?= /tools/Xilinx/Vitis/2021.2

HOST_CXX ?= aarch64-linux-gnu-g++
RM = rm -f
RMDIR = rm -rf

VITIS_PLATFORM = zcu102_base
VITIS_PLATFORM_DIR = /media/SSD2TB/wd/vitis_ide_projects/zcu102_base/export/zcu102_base
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/zcu102_base.xpfm

# host compiler global settings
CXXFLAGS += -std=c++1y -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I/tools/Xilinx/Vitis_HLS/2021.2/include/ -I/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux/usr/include/xrt/ -O2 -g -Wall -c -fmessage-length=0 --sysroot=/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux
LDFLAGS += -luuid -lxrt_coreutil -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/ --sysroot=/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux

#
# host files
#

HOST_OBJECTS += src/fibha_host.o
HOST_OBJECTS += src/prepare_weights_and_input.o
HOST_OBJECTS += src/test_utils.o

HOST_EXE = fibha_v1

BUILD_SUBDIRS += src/

#
# primary build targets
#

.PHONY: all clean
all:  $(HOST_EXE)

clean:
	-$(RM) $(HOST_EXE) $(HOST_OBJECTS)

.PHONY: incremental
incremental: all


nothing:

#
# host rules
#

src/fibha_host.o: ../src/fibha_host.cpp ../src/fibha_host.h ../../fibha_v1_kernels/src/all_includes.h ../../../../../../tools/Xilinx/Vitis_HLS/2021.2/include/ap_int.h ../../../../../..$(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/aarch64-xilinx-linux/usr/include/c++/10.2.0/cstdlib ../src/prepare_weights_and_inputs.h ../src/test_utils.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/prepare_weights_and_input.o: ../src/prepare_weights_and_input.cpp ../src/prepare_weights_and_inputs.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/test_utils.o: ../src/test_utils.cpp ../src/test_utils.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)

