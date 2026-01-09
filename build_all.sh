export eval_root=`pwd`

cd ads_int
make -j8
cd $eval_root

cd PaRMAT/Release
make -j8
cd $eval_root

git clone https://github.com/IntelligentSoftwareSystems/Galois.git nf_int
cd nf_int
git checkout 38cd91cfb59a30cf0b4f7cf7d19f29d4d7188548
git apply ../nf_int.patch
git submodule init
git submodule update
mkdir build
cd build/
cmake .. -DGALOIS_CUDA_CAPABILITY="7.5" 
cd lonestar/analytics/gpu/sssp/
make -j8
cd $eval_root