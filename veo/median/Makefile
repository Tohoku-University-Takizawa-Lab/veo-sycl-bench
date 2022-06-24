include ../config.mk

TARGET = median kernel.so
all: $(TARGET)

median: median.o
	$(CXX) $< -o $@ $(VEFLAGS)
median.o: median.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./median $(PROBLEM_SIZE)
