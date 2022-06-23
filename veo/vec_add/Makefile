include ../config.mk

TARGET = vec_add kernel.so
all: $(TARGET)

vec_add: vec_add.o
	$(CXX) $< -o $@ $(VEFLAGS)
vec_add.o: vec_add.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./vec_add $(PROBLEM_SIZE)
