export CUDA_VISIBLE_DEVICES=5,6

python bin/iterate_davis2011kinase.py gp perprot > iterate_davis2011kinase_gp_perprot.log 2>&1 &

python bin/iterate_davis2011kinase.py mlper1 perprot > iterate_davis2011kinase_mlper1_perprot.log 2>&1
python bin/iterate_davis2011kinase.py mlper1g perprot > iterate_davis2011kinase_mlper1g_perprot.log 2>&1
python bin/iterate_davis2011kinase.py mlper5 perprot > iterate_davis2011kinase_mlper5_perprot.log 2>&1
python bin/iterate_davis2011kinase.py mlper5g perprot > iterate_davis2011kinase_mlper5g_perprot.log 2>&1
python bin/iterate_davis2011kinase.py hybrid perprot > iterate_davis2011kinase_hybrid_perprot.log 2>&1

python bin/parse_log.py gp perprot >> iterate_davis2011kinase_gp_perprot.log 2>&1
python bin/parse_log.py mlper1 perprot >> iterate_davis2011kinase_mlper1_perprot.log 2>&1
python bin/parse_log.py mlper1g perprot >> iterate_davis2011kinase_mlper1g_perprot.log 2>&1
python bin/parse_log.py mlper5 perprot >> iterate_davis2011kinase_mlper5_perprot.log 2>&1
python bin/parse_log.py mlper5g perprot >> iterate_davis2011kinase_mlper5g_perprot.log 2>&1
python bin/parse_log.py hybrid perprot >> iterate_davis2011kinase_hybrid_perprot.log 2>&1

wait
