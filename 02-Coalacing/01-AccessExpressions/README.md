# Analysis

* Access aligend will ensure the fastest memory access as the memory acces is nicely coalaced and only one data read per warp happens.
* Access with offset can miss the warp allignement causing global read twice per warp.
* Access with stride is much worse as it can cause multiple global reads per warp.

### Logs
```
root@teja:~/Projs/CUDA/02-Coalacing/01-AccessExpressions# ./make_run.sh 
--metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,
make app
make[1]: Entering directory '/root/Projs/CUDA/02-Coalacing/01-AccessExpressions'
nvcc  -I. -I../infra -dlink -dc main.cu 
nvcc  *.o    -o run 
make[1]: Leaving directory '/root/Projs/CUDA/02-Coalacing/01-AccessExpressions'
==PROF== Connected to process 17225 (/root/Projs/CUDA/02-Coalacing/01-AccessExpressions/run)
==PROF== Profiling "alligned_access" - 1: 0%....50%....100% - 1 pass
==PROF== Disconnected from process 17225
[17225] run@127.0.0.1
  alligned_access(float*,int), 2020-May-09 16:47:12, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                           1.04
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                           1.04
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                         16,384
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         16,384
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 17257 (/root/Projs/CUDA/02-Coalacing/01-AccessExpressions/run)
==PROF== Profiling "offset_access" - 1: 0%....50%....100% - 1 pass
==PROF== Disconnected from process 17257
[17257] run@127.0.0.1
  offset_access(float*,int,int), 2020-May-09 16:47:13, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                           1.20
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                           1.40
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                         17,568
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         20,480
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 17293 (/root/Projs/CUDA/02-Coalacing/01-AccessExpressions/run)
==PROF== Profiling "strided_access" - 1: 0%....50%....100% - 1 pass
==PROF== Disconnected from process 17293
[17293] run@127.0.0.1
  strided_access(float*,int,int), 2020-May-09 16:47:14, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                           1.75
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Gbyte/second                           1.75
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                         32,768
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         32,768
    ---------------------------------------------------------------------- --------------- ------------------------------
```