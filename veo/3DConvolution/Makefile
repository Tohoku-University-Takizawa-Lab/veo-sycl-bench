include ../config.mk

TARGET = 3DConvolution kernel.so
all: $(TARGET)

3DConvolution: 3DConvolution.o
	$(CXX) $< -o $@ $(VEFLAGS)
3DConvolution.o: 3DConvolution.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./3DConvolution $(PROBLEM_SIZE)
