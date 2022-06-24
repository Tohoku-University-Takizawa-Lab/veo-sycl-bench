SUBDIRS := $(wildcard */.)

.PHONY: all clean run

all clean run:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
