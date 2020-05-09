CXX := g++ -std=c++11 
NVCC := nvcc 

LDFLAGS += $(shell pkg-config opencv --cflags --libs)

all:
	$(CXX) showImage.cpp -o showImage $(LDFLAGS)
	$(NVCC) image_blur.cu -o run $(LDFLAGS)
clean:
	rm -f showImage
	rm -f run
	rm -f result.jpg