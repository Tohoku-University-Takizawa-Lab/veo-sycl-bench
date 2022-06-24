VEO_HEADER_PATH := /opt/nec/ve/veos/include
VEO_LIB_PATH := /opt/nec/ve/veos/lib64
VEFLAGS := -I../../include -I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC := /opt/nec/ve/bin/ncc
CXX := g++

PROBLEM_SIZE := 100
