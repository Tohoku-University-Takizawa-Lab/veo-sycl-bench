NEO_SYCL = /home/gp.sc.cc.tohoku.ac.jp/lijiahao/workspace/neoSYCL-2021/include
VEO_HEADER_PATH=/opt/nec/ve/veos/include
VEO_LIB_PATH=/opt/nec/ve/veos/lib64

CXX = g++
# CXXFLAGS = -std=c++17 -DDEBUG 
CXXFLAGS = -std=c++17
VEFLAGS = -I ../include  -I$(NEO_SYCL) -I$(VEO_HEADER_PATH) -lpthread -L$(VEO_LIB_PATH) -Wl,-rpath=$(VEO_LIB_PATH) -lveo

NCC = /opt/nec/ve/bin/ncc
C = gcc
SOURCE = nbody.cpp 2DConvolution.cpp 2mm.cpp 3DConvolution.cpp 3mm.cpp atax.cpp bicg.cpp correlation.cpp covariance.cpp fdtd2d.cpp gemm.cpp gesummv.cpp kmeans.cpp median.cpp mol_dyn mvt.cpp scalar_prod.cpp sobel.cpp sobel5.cpp sobel7.cpp syrk.cpp syrk2.cpp vec_add.cpp lin_reg_coeff.cpp lin_reg_error.cpp
TARGET = nbody 2DConvolution 2mm 3DConvolution 3mm atax bicg correlation covariance fdtd2d gemm gesummv kmeans median mol_dyn mvt scalar_prod sobel sobel5 sobel7 syrk syrk2 vec_add lin_reg_coeff lin_reg_error
KERNEL = kernel.so 

all:$(TARGET) ${KERNEL} ${VEO}

nbody:nbody.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
2DConvolution:2DConvolution.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
2mm:2mm.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
3DConvolution:3DConvolution.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
3mm:3mm.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
atax:atax.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
bicg:bicg.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
correlation:correlation.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
covariance:covariance.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
fdtd2d:fdtd2d.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
gemm:gemm.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
gesummv:gesummv.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
kmeans:kmeans.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
median:median.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
mol_dyn:mol_dyn.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
mvt:mvt.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
scalar_prod:scalar_prod.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
sobel:sobel.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
sobel5:sobel5.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
sobel7:sobel7.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
syrk:syrk.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
syrk2:syrk2.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
vec_add:vec_add.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
lin_reg_coeff:lin_reg_coeff.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)
lin_reg_error:lin_reg_error.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(VEFLAGS)

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(VEFLAGS)

$(KERNEL): kernel.c
	$(NCC) -fpic -shared -o $@ $<

	
.PHONY:clean 
clean:
	-rm -rf $(TARGET) $(obj)  