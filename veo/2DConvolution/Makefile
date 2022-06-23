include ../config.mk

TARGET = 2DConvolution kernel.so
all:$(TARGET)

2DConvolution: 2DConvolution.o
	$(CXX) $< -o $@ $(VEFLAGS)
2DConvolution.o: 2DConvolution.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so:kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./2DConvolution $(PROBLEM_SIZE)
