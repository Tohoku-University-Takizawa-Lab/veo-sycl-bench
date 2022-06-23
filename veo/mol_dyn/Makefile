include ../config.mk

TARGET = mol_dyn kernel.so
all: $(TARGET)

mol_dyn: mol_dyn.o
	$(CXX) $< -o $@ $(VEFLAGS)
mol_dyn.o: mol_dyn.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./mol_dyn $(PROBLEM_SIZE)
