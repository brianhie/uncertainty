export CUDA_VISIBLE_DEVICES=2

python bin/iterate_davis2011kinase.py gp partition 10 > iterate_davis2011kinase_gp_partition.log 2>&1 &

source activate edward
python bin/iterate_davis2011kinase.py bayesnn partition 10 > iterate_davis2011kinase_bayesnn_partition.log 2>&1 &
conda deactivate

python bin/iterate_davis2011kinase.py mlper1 partition 10 > iterate_davis2011kinase_mlper1_partition.log 2>&1
python bin/iterate_davis2011kinase.py mlper1g partition 10 > iterate_davis2011kinase_mlper1g_partition.log 2>&1
python bin/iterate_davis2011kinase.py mlper5g partition 10 > iterate_davis2011kinase_mlper5g_partition.log 2>&1
python bin/iterate_davis2011kinase.py hybrid partition 10 > iterate_davis2011kinase_hybrid_partition.log 2>&1

wait

python bin/parse_log.py gp partition >> iterate_davis2011kinase_gp_partition.log 2>&1
python bin/parse_log.py bayesnn partition >> iterate_davis2011kinase_bayesnn_partition.log 2>&1
python bin/parse_log.py mlper1 partition >> iterate_davis2011kinase_mlper1_partition.log 2>&1
python bin/parse_log.py mlper1g partition >> iterate_davis2011kinase_mlper1g_partition.log 2>&1
python bin/parse_log.py mlper5g partition >> iterate_davis2011kinase_mlper5g_partition.log 2>&1
python bin/parse_log.py hybrid partition >> iterate_davis2011kinase_hybrid_partition.log 2>&1
