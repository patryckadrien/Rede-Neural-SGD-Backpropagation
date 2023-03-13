FC = g++
FFLAGS = -O2 -fopenmp -fexpensive-optimizations -m64 -foptimize-register-move -funroll-loops -ffast-math
#FC = ifort
#FFLAGS = -O2 -qopenmp -mp1 -zero -xHOST -ipo -align 
#array64byte
#FFLAGS = -O2 -qopenmp -xW -align -fno-alias
OBJ = sgdbpp.o

%.o: %.cpp
	$(FC) $(FFLAGS) -c -o $@ $<

sgdbpp: $(OBJ)
	$(FC) $(FFLAGS) -o sgdbpp $(OBJ)

.PHONY:clean

clean:
	rm sgdbpp *.out  *.o
