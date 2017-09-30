# Makefile for TUPU Exam Code
.PHONY: all clean

# Build tools and flags
LD         := $(CXX)
CPPFLAGS   := -I. -Ilayers
CXXFLAGS   := -std=c++11
LDFLAGS    := 
LDLIBS     := 

# OpenCV... If you want to use predict feature, Please set 0 to 1
# We just use OpenCV to read non-mnist images.
OPENCV_ENABLE = 0

ifeq ($(OPENCV_ENABLE), 1)
CPPFLAGS += -D WITH_OPENCV=1
CPPFLAGS += $(shell pkg-config --cflags opencv)
LDLIBS += $(shell pkg-config --libs opencv)
endif


# Rules for building
all: main

main:main.cpp net.o layers/convLayer.o \
 layers/dataLayer.o layers/tanhLayer.o \
 layers/poolingLayer.o layers/fcLayer.o \
 layers/lossLayer.o layers/fcLayer.o  layers/imageInputLayer.o 

net.o: net.cpp net.h

convLayer.o: layers/convLayer.*
dataLayer.o: layers/dataLayer.*
tanhLayer.o: layers/tanhLayer.*
poolingLayer.o: layers/poolingLayer.*
fcLayer.o: layers/fcLayer.*
lossLayer.o: layers/lossLayer.*
imageInputLayer.o: layers/imageInputLayer.*

clean:
	$(RM) *.o layers/*.o main
