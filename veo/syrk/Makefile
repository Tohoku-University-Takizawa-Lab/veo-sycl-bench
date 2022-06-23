include ../config.mk

TARGET = syrk kernel.so
all: $(TARGET)

syrk: syrk.o
	$(CXX) $< -o $@ $(VEFLAGS)
syrk.o: syrk.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./syrk $(PROBLEM_SIZE)
