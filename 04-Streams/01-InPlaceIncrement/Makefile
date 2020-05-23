NVCC := nvcc 
MAKE ?= make

LIBS := #-lcurand

NVCCFLAGS := -I. \
			 -I../infra \
			 -dlink \
			 -lineinfo 

BUILT_LIBS := 

LDFLAGS += #$(shell pkg-config opencv --cflags --libs)

OBJS := *.o 

DEFS := #-DUNIT_TEST
		

all:
	$(MAKE) app

app:
	$(NVCC) $(DEFS) $(NVCCFLAGS) -dc *.cu $(LDFLAGS)
	$(NVCC) main.o $(LIBS) $(BUILT_LIBS) -o run $(LDFLAGS)

clean:
	rm -f run
	rm -f *.o
	rm -f *.a
