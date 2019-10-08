export CUDA_VISIBLE_DEVICES=3,4

python bin/iterate_davis2011kinase.py gp explore 10 > iterate_davis2011kinase_gp_explore.log 2>&1 &

source activate edward
python bin/iterate_davis2011kinase.py bayesnn explore 10 > iterate_davis2011kinase_bayesnn_explore.log 2>&1 &
conda deactivate

python bin/iterate_davis2011kinase.py mlper1 explore 10 > iterate_davis2011kinase_mlper1_explore.log 2>&1
python bin/iterate_davis2011kinase.py mlper1g explore 10 > iterate_davis2011kinase_mlper1g_explore.log 2>&1
python bin/iterate_davis2011kinase.py mlper5g explore 10 > iterate_davis2011kinase_mlper5g_explore.log 2>&1
python bin/iterate_davis2011kinase.py hybrid explore 10 > iterate_davis2011kinase_hybrid_explore.log 2>&1

wait

python bin/parse_log.py gp explore >> iterate_davis2011kinase_gp_explore.log 2>&1
python bin/parse_log.py bayesnn explore >> iterate_davis2011kinase_bayesnn_explore.log 2>&1
python bin/parse_log.py mlper1 explore >> iterate_davis2011kinase_mlper1_explore.log 2>&1
python bin/parse_log.py mlper1g explore >> iterate_davis2011kinase_mlper1g_explore.log 2>&1
python bin/parse_log.py mlper5g explore >> iterate_davis2011kinase_mlper5g_explore.log 2>&1
python bin/parse_log.py hybrid explore >> iterate_davis2011kinase_hybrid_explore.log 2>&1
