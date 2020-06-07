#!/bin/bash

make all

./run

#FOR DEBUG


# nvsightFlags='--metrics '
# nvsightFlags+='l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,'
# nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
# nvsightFlags+='l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,'
# nvsightFlags+='l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,'
# nvsightFlags+='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,'
# nvsightFlags+='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second,'
# nvsightFlags+='gpu__time_active.sum,'
# nvsightFlags+=' '
# #nvsightFlags+='--set full '
# nvsightFlags+='--open-in-ui '

# nvsightFlags+='--set default '
# #nvsightFlags+='-o profile '


# echo $nvsightFlags


#nv-nsight-cu-cli $nvsightFlags -f ./run
#cuda-memcheck ./run
#cuobjdump -sass mycode

make clean


