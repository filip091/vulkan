CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkanTest: main.cpp
	g++ $(CFLAGS) -o vulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: vulkanTest
	./vulkanTest

clean:
	rm -f vulkanTest
