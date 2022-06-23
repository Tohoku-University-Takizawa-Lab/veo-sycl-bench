include ../config.mk

TARGET = sobel7 kernel.so
all: $(TARGET)

sobel7: sobel7.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel7.o: sobel7.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./sobel7 $(PROBLEM_SIZE)
