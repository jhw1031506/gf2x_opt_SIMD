cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=0 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=1 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=2 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=3 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=0 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=1 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=1 -DGFMUL_VER=2 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=0 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=1 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=2 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=3 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=0 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=1 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=2 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=5 -DGFMUL_VER=0 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=5 -DGFMUL_VER=1 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=5 -DGFMUL_VER=3 -DVPCLMUL=0"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=5 -DGFMUL_VER=0 -DVPCLMUL=1"  .. &&\
make && \
./bike-test && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=5 -DGFMUL_VER=1 -DVPCLMUL=1"  .. &&\
make && \
./bike-test