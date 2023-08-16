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
LDFLAGS += -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/ -shared --sysroot=/opt/petalinux/2021.2//sysroots/cortexa72-cortexa53-xilinx-linux

#
# host files
#

HOST_OBJECTS += src/client/cpp_fc.o
HOST_OBJECTS += src/client/prepare_weights_and_input.o
HOST_OBJECTS += src/client/prepare_weights_and_input_v2.o
HOST_OBJECTS += src/fibha_host.o
HOST_OBJECTS += src/main_tester.o
HOST_OBJECTS += src/tests/main_tester.o
HOST_OBJECTS += src/tests/test_utils.o

HOST_EXE = fiba_v2

BUILD_SUBDIRS += src/client/
BUILD_SUBDIRS += src/
BUILD_SUBDIRS += src/tests/

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

src/client/cpp_fc.o: ../src/client/cpp_fc.cpp ../src/client/cpp_fc.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/client/prepare_weights_and_input.o: ../src/client/prepare_weights_and_input.cpp ../src/client/prepare_weights_and_inputs.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/client/prepare_weights_and_input_v2.o: ../src/client/prepare_weights_and_input_v2.cpp ../src/client/prepare_weights_and_inputs.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/fibha_host.o: ../src/fibha_host.cpp ../../../../../../tools/Xilinx/Vitis_HLS/2021.2/include/ap_int.h ../../../../../../usr/include/c++/7/cstdlib ../src/fibha_host.h ../src/client/prepare_weights_and_inputs.h ../src/tests/test_utils.h ../src/client/cpp_fc.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/main_tester.o: ../src/main_tester.cpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/tests/main_tester.o: ../src/tests/main_tester.cpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/tests/test_utils.o: ../src/tests/test_utils.cpp ../src/tests/test_utils.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o lib"$@" $(+) $(LDFLAGS)

