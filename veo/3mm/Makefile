include ../config.mk

TARGET = 3mm kernel.so
all: $(TARGET)

3mm: 3mm.o
	$(CXX) $< -o $@ $(VEFLAGS)
3mm.o: 3mm.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./3mm $(PROBLEM_SIZE)
