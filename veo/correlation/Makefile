include ../config.mk

TARGET = correlation kernel.so
all: $(TARGET)

correlation: correlation.o
	$(CXX) $< -o $@ $(VEFLAGS)
correlation.o: correlation.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./correlation $(PROBLEM_SIZE)
