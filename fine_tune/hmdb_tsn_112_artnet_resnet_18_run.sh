#!/usr/bin/env sh

TOOLS=/home/lmwang/code/c3d_caffe/caffe/cmake_build_c3d/install/bin

nohup /usr/local/openmpi/bin/mpirun -n 8 \
$TOOLS/caffe train --solver=hmdb_tsn_112_artnet_resnet_18_solver.prototxt --weights=112_artnet_tsn_resnet_18_kinetics.caffemodel  > hmdb51_split1_tsn_artnet_seg_2_log_0.out &
    
echo "Done."

