export eval_root=`pwd`

rm *_result

cd ads_int
make clean
cd $eval_root

cd nv_int
make clean
cd $eval_root
