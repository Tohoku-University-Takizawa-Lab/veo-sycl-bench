include ../config.mk

TARGET = mvt kernel.so
all: $(TARGET)

mvt: mvt.o
	$(CXX) $< -o $@ $(VEFLAGS)
mvt.o: mvt.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./mvt $(PROBLEM_SIZE)
