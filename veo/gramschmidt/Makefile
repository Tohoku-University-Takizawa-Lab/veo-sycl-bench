include ../config.mk

TARGET = gramschmidt kernel.so
all: $(TARGET)

gramschmidt: gramschmidt.o
	$(CXX) $< -o $@ $(VEFLAGS)
gramschmidt.o: gramschmidt.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./gramschmidt $(PROBLEM_SIZE)
