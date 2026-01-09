A=${1:-0.57}
B=${2:-0.19}
C=${3:-0.19}

./PaRMAT/Release/PaRMAT -nVertices 100000000 -nEdges 1000000000 -a $A -b $B -c $C -noEdgeToSelf -noDuplicateEdges -threads 30 -output ./PaRMAT/Release/out.txt
./nf_int/build/tools/graph-convert/graph-convert --mtx2gr --edgeType=int32 ./PaRMAT/Release/out.txt ./my_data/graph_gr/tmp.gr > /dev/null
rm ./PaRMAT/Release/out.txt

./ads_int_pro/sssp ./my_data/graph_gr/tmp.gr -v -s 0 3>&1 -o ./ads_int_pro_result_vr > ./logs/ads_int_pro_vr_log
./ads_int_pro/sssp ./my_data/graph_gr/tmp.gr -s 0 3>&1 -o ./ads_int_pro_result_novr > ./logs/ads_int_pro_novr_log
./ads_int/sssp ./my_data/graph_gr/tmp.gr -s 0 3>&1 -o ./ads_int_result > ./logs/ads_int_log
./ads_mine/sssp ./my_data/graph_gr/tmp.gr -s 0 3>&1 -o ./ads_mine_result > ./logs/ads_mine_log

rm ./my_data/graph_gr/tmp.gr

if ! diff -q ./ads_int_pro_result_vr ./ads_mine_result > /dev/null; then
    echo "Difference found between ads_int_pro_result_vr and ads_mine_result"
fi

if ! diff -q ./ads_int_pro_result_novr ./ads_mine_result > /dev/null; then
    echo "Difference found between ads_int_pro_result_novr and ads_mine_result"
fi

if ! diff -q ./ads_int_result ./ads_mine_result > /dev/null; then
    echo "Difference found between ads_int_result and ads_mine_result"
fi

rm ./ads_int_pro_result_vr ./ads_int_pro_result_novr ./ads_int_result ./ads_mine_result