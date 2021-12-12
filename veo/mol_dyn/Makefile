VEO_HEADER_PATH=/opt/nec/ve/veos/include
VEO_LIB_PATH=/opt/nec/ve/veos/lib64
VEFLAGS=-I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC=/opt/nec/ve/bin/ncc

CXX=g++
# CXXFLAGS = -std=c++17 -DDEBUG 

TARGET = mol_dyn kernel.so
all:$(TARGET)

mol_dyn:mol_dyn.o
	$(CXX) $< -o $@ $(VEFLAGS)
mol_dyn.o:mol_dyn.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so:kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)	