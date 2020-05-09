#!/bin/bash


nvsightFlags='--metrics'
nvsightFlags+=' l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,'
nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
nvsightFlags+='l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,'
nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'
nvsightFlags+=' --set full'
nvsightFlags+=' --open-in-ui'


echo $nvsightFlags

make all
nv-nsight-cu-cli $nvsightFlags -f ./run
make clean