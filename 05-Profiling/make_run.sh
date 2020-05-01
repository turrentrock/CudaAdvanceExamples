#!/bin/bash

make all
nv-nsight-cu-cli -o profile_Transpose_rowRead_colWrite -f ./run 0
nv-nsight-cu-cli -o profile_Transpose_colRead_rowWrite -f ./run 1
nv-nsight-cu profile_Transpose_rowRead_colWrite.nsight-cuprof-report
nv-nsight-cu profile_Transpose_colRead_rowWrite.nsight-cuprof-report


make clean