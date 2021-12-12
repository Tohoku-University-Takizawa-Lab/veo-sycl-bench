VEO_HEADER_PATH=/opt/nec/ve/veos/include
VEO_LIB_PATH=/opt/nec/ve/veos/lib64
VEFLAGS=-I ../../include -I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC=/opt/nec/ve/bin/ncc

CXX=g++
# CXXFLAGS = -std=c++17 -DDEBUG 

TARGET = sobel3 kernel.so
all:$(TARGET)

sobel3:sobel3.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel3.o:sobel3.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so:kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)	