# Example Makefile. Before compile,
#load module courses/cpsc8200
CC = mpicc
CLINKER = mpicc
F77 = mpif77
CFLAGS = -O3
FFLAGS = -O3
MATH_LIB = -lm

all: build

build: summa summa_test cartesian

summa: summa.c
	$(CLINKER) -o summa summa.c $(MATH_LIB)
summa_test: summa_test.c
	$(CLINKER) -o summa_test summa_test.c $(MATH_LIB)
cartesian: cartesian.c
	$(CLINKER) -o cartesian cartesian.c $(MATH_LIB)
.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<


# .c.o:
# 	$(CC) $(CFLAGS) -c $<
# .f.o:
# 	$(F77) $(FFLAGS) -c $<

# cartesian: cartesian.c
# 	$(CLINKER) -o cartesian cartesian.c $(MATH_LIB)
# .c.o:
# 	$(CC) $(CFLAGS) -c $<
# .f.o:
# 	$(F77) $(FFLAGS) -c $<

# clean:
# 	rm -f *.o summa cartesian

