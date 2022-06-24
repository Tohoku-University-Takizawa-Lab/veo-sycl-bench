include ../config.mk

TARGET = 2mm kernel.so
all: $(TARGET)

2mm: 2mm.o
	$(CXX) $< -o $@ $(VEFLAGS)
2mm.o: 2mm.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./2mm $(PROBLEM_SIZE)
