NVCC := nvcc 
MAKE ?= make

LIBS := #-lcurand

NVCCFLAGS := -I. \
			 -I../infra \
			 -dlink

BUILT_LIBS := 

LDFLAGS += #$(shell pkg-config opencv --cflags --libs)

OBJS := *.o 

DEFS := #-DUNIT_TEST

all:
	$(MAKE) cpu_app
	$(MAKE) app

cpu_app:
	$(NVCC) $(DEFS) main.cpp -o cpu_version

app:
	$(NVCC) $(DEFS) $(NVCCFLAGS) -dc main.cu $(LDFLAGS)
	$(NVCC) $(OBJS) $(LIBS) $(BUILT_LIBS) -o gpu_version $(LDFLAGS)

clean:
	rm -f *_version
	rm -f *.o
	rm -f *.a
