SOURCE = face_recognition.cpp function.cpp
TARGET = mobilefacenet.elf
CXX = g++
CXXFLAGS = -Wall -std=c++11
LIBS = -lm

$(TARGET):$(SOURCE)
	$(CXX) $^ -o $@ `pkg-config opencv --cflags --libs`

clean:
	rm $(TARGET)
