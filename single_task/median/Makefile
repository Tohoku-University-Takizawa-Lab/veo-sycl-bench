VEO_HEADER_PATH=/opt/nec/ve/veos/include
VEO_LIB_PATH=/opt/nec/ve/veos/lib64
NEO_SYCL = /home/gp.sc.cc.tohoku.ac.jp/lijiahao/workspace/neoSYCL-2021/include

VEFLAGS=-I../../include -I$(NEO_SYCL) -I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC=/opt/nec/ve/bin/ncc

CXX=g++
# CXXFLAGS = -std=c++17 -DDEBUG 
CXXFLAGS = -std=c++17

TARGET = median kernel.so
all:$(TARGET)

median:median.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
median.o:median.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(VEFLAGS)
kernel.so:kernel.c
	$(NCC) -fpic -shared -o $@ $< -fopenmp

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)	