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
#nvsightFlags+='--open-in-ui '

#nvsightFlags+='--set default '
#nvsightFlags+='-o profile '


echo $nvsightFlags

make all
nv-nsight-cu-cli $nvsightFlags -f ./run 0
nv-nsight-cu-cli $nvsightFlags -f ./run 1
nv-nsight-cu-cli $nvsightFlags -f ./run 2
nv-nsight-cu-cli $nvsightFlags -f ./run 3
nv-nsight-cu-cli $nvsightFlags -f ./run 4
nv-nsight-cu-cli $nvsightFlags -f ./run 5
make clean