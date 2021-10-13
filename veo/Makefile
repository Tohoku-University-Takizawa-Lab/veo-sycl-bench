VEO_HEADER_PATH=/opt/nec/ve/veos/include
VEO_LIB_PATH=/opt/nec/ve/veos/lib64
VEFLAGS = -I ../include -I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC = /opt/nec/ve/bin/ncc
C = gcc

CXX = g++
CXXFLAGS = -std=c++17 -DDEBUG 

SOURCE = 2mm.c 3mm.c 2DConvolution.c 3DConvolution.c atax.c bicg.c correlation.c covariance.c fdtd2d.c gemm.c gesummv.c gramschmidt.c mvt.c syrk.c syrk2.c vec_add.c scalar_prod.c mol_dyn.c lin_reg_coeff.c kmeans.c nbody.c lin_reg_error.c
TARGET = 2mm 3mm 2DCon 3DCon atax bicg correlation covariance fdtd2d gemm gesummv gramschmidt mvt syrk syrk2 vec_add scalar_prod mol_dyn lin_reg_coeff kmeans nbody median sobel sobel5 sobel7 lin_reg_error
# OBJ = 2mm.o 3mm.o 2DConvolution.o 3DConvolution.o atax.o bicg.o correlation.o covariance.o fdtd2d.o gemm.o gesummv.o gramschmidt.o mvt.o syrk.o syrk2.o vec_add.o scalar_prod.o
KERNEL = kernel.so
all:$(TARGET) ${KERNEL}

2mm:2mm.o
	$(C) $< -o $@ $(VEFLAGS)
3mm:3mm.o
	$(C) $< -o $@ $(VEFLAGS)
2DCon:2DConvolution.o
	$(C) $< -o $@ $(VEFLAGS)
3DCon:3DConvolution.o
	$(C) $< -o $@ $(VEFLAGS)
atax:atax.o
	$(C) $< -o $@ $(VEFLAGS)
bicg:bicg.o
	$(C) $< -o $@ $(VEFLAGS)
correlation:correlation.o
	$(C) $< -o $@ $(VEFLAGS)
covariance:covariance.o
	$(C) $< -o $@ $(VEFLAGS)
fdtd2d:fdtd2d.o
	$(C) $< -o $@ $(VEFLAGS)
gemm:gemm.o
	$(C) $< -o $@ $(VEFLAGS)
gesummv:gesummv.o
	$(C) $< -o $@ $(VEFLAGS)
gramschmidt:gramschmidt.o
	$(C) $< -o $@ $(VEFLAGS)
mvt:mvt.o
	$(C) $< -o $@ $(VEFLAGS)
syrk:syrk.o
	$(C) $< -o $@ $(VEFLAGS)
syrk2:syrk2.o
	$(C) $< -o $@ $(VEFLAGS)
vec_add:vec_add.o
	$(C) $< -o $@ $(VEFLAGS)
scalar_prod:scalar_prod.o
	$(C) $< -o $@ $(VEFLAGS)
mol_dyn:mol_dyn.o
	$(C) $< -o $@ $(VEFLAGS)
lin_reg_coeff:lin_reg_coeff.o
	$(C) $< -o $@ $(VEFLAGS)
kmeans:kmeans.o
	$(C) $< -o $@ $(VEFLAGS)
nbody:nbody.o
	$(C) $< -o $@ $(VEFLAGS)
median:median.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel:sobel.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel5:sobel5.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel7:sobel7.o
	$(CXX) $< -o $@ $(VEFLAGS)
lin_reg_error:lin_reg_error.o
	$(CXX) $< -o $@ $(VEFLAGS)

%.o:%.c
	$(C) -c $< -o $@ $(VEFLAGS)
median.o:median.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
sobel.o:sobel.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
sobel5.o:sobel5.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
sobel7.o:sobel7.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)

$(KERNEL): kernel.c
	$(NCC) -std=c99 -fpic -shared -o $@ $<
	
.PHONY:clean 
clean:
	-rm -rf $(TARGET) $(obj)  