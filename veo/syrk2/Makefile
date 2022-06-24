include ../config.mk

TARGET = syrk2 kernel.so
all: $(TARGET)

syrk2: syrk2.o
	$(CXX) $< -o $@ $(VEFLAGS)
syrk2.o: syrk2.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./syrk2 $(PROBLEM_SIZE)
