include ../config.mk

TARGET = gemm kernel.so
all: $(TARGET)

gemm: gemm.o
	$(CXX) $< -o $@ $(VEFLAGS)
gemm.o: gemm.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./gemm $(PROBLEM_SIZE)
