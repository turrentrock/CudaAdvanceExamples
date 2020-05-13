# Analysis



### Logs
```
==PROF== Connected to process 13540 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v0" - 1: 0%....50%....100% - 3 passes
Time for v0: 0.20447 seconds
==PROF== Disconnected from process 13540
[13540] run@127.0.0.1
  reduce_v0(float*,float*,int), 2020-May-13 18:54:24, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                          13.25
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                     1,89,49,18,839.25
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                     1,10,33,07,835.91
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          20.26
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         158.32
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         65,537
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 13572 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v1" - 1: 0%....50%....100% - 3 passes
Time for v1: 0.189387 seconds
==PROF== Disconnected from process 13572
[13572] run@127.0.0.1
  reduce_v1(float*,float*,int), 2020-May-13 18:54:24, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                           8.31
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                     3,02,00,08,888.94
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                     1,75,83,86,376.59
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          32.30
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         252.32
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         65,537
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 13610 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v2" - 1: 0%....50%....100% - 3 passes
Time for v2: 0.168808 seconds
==PROF== Disconnected from process 13610
[13610] run@127.0.0.1
  reduce_v2(float*,float*,int), 2020-May-13 18:54:25, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                           7.39
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                       64,77,91,022.51
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                       60,34,21,774.40
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          36.35
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         283.96
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         65,537
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 13642 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v3" - 1: 0%....50%....100% - 3 passes
Time for v3: 0.152817 seconds
==PROF== Disconnected from process 13642
[13642] run@127.0.0.1
  reduce_v3(float*,float*,int), 2020-May-13 18:54:26, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                           3.78
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                       63,25,56,662.41
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                       58,92,30,863.61
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          70.98
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         277.29
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,864
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         32,769
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 13674 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v4" - 1: 0%....50%....100% - 3 passes
Time for v4: 0.147517 seconds
==PROF== Disconnected from process 13674
[13674] run@127.0.0.1
  reduce_v4(float*,float*,int), 2020-May-13 18:54:27, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                           2.64
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                       90,55,96,584.37
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                       84,35,69,421.06
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         101.63
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         396.97
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,864
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         32,769
    ---------------------------------------------------------------------- --------------- ------------------------------

==PROF== Connected to process 13706 (/root/Projs/CUDA/03-Reduction/run)
==PROF== Profiling "reduce_v5" - 1: 0%....50%....100% - 3 passes
Time for v5: 0.152817 seconds
==PROF== Disconnected from process 13706
[13706] run@127.0.0.1
  void reduce_v5<unsigned int=1024>(float*,float*,int), 2020-May-13 18:54:28, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    gpu__time_active.sum                                                           msecond                           2.38
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second                                     1,00,50,45,526.96
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second                                       93,62,06,792.24
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                         112.79
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second                   Mbyte/second                         440.57
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      83,88,864
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                                  sector                         32,769
    ---------------------------------------------------------------------- --------------- ------------------------------

```