#!/bin/bash


nvsightFlags='--metrics '
nvsightFlags+='l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,'
nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
nvsightFlags+='l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,'
nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,'
nvsightFlags+='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,'
nvsightFlags+='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second,'
nvsightFlags+='gpu__time_active.sum,'
nvsightFlags+=' '
#nvsightFlags+='--set full '
nvsightFlags+='--open-in-ui '

nvsightFlags+='--set default '
#nvsightFlags+='-o profile '


echo $nvsightFlags

make all

./run 1
./run 2
./run 4
./run 8
./run 16
#FOR DEBUG

#nv-nsight-cu-cli $nvsightFlags -f ./run 8
#cuda-memcheck ./run

make clean


