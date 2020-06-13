NVCC := nvcc 
MAKE ?= make

LIBS := #-lcurand

NVCCFLAGS := -I. \
			 -I../infra \
			 -dlink \
			 -lineinfo \

BUILT_LIBS := 

LDFLAGS += 

OBJS := *.o 

DEBUG := 	-ptx \
			-src-in-ptx

DEBUG :=		

all:
	$(MAKE) app

app:
	$(NVCC) $(DEBUG) $(NVCCFLAGS) -dc *.cu $(LDFLAGS)
	$(NVCC) main.o $(LIBS) $(BUILT_LIBS) -o run $(LDFLAGS)

clean:
	rm -f run
	rm -f *.o
	rm -f *.a
	rm -f *.ptx
