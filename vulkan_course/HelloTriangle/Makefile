VULKAN_SDK_PATH = /usr/include/vulkan
STD_INCLUDE_PATH = /home/filip0913/projects/stb-master

CFLAGS = -std=c++17 -O2 -I$(VULKAN_SDK_PATH)/include -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkanTest: main.cpp
	g++ $(CFLAGS) -o vulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: vulkanTest
	./vulkanTest

clean:
	rm -f vulkanTest
