PROGRAM=pcolpicker
FILES=src/pcolpicker.cpp

CC=g++
CXXFLAGS=-I/usr/local/include
LDFLAGS=-L/usr/local/lib
LIBS=-lopencv_core -lopencv_highgui

.SUFFIXES: .cpp .o

$(PROGRAM): $(FILES)
	$(CC) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $(PROGRAM) $<

.PHONY: clean
clean:
	rm -f $(PROGRAM)

