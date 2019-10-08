export CUDA_VISIBLE_DEVICES=3,4

python bin/iterate_davis2011kinase.py gp quad > iterate_davis2011kinase_gp_quad.log 2>&1 &

python bin/iterate_davis2011kinase.py mlper1 quad > iterate_davis2011kinase_mlper1_quad.log 2>&1
python bin/iterate_davis2011kinase.py mlper1g quad > iterate_davis2011kinase_mlper1g_quad.log 2>&1
python bin/iterate_davis2011kinase.py mlper5 quad > iterate_davis2011kinase_mlper5_quad.log 2>&1
python bin/iterate_davis2011kinase.py mlper5g quad > iterate_davis2011kinase_mlper5g_quad.log 2>&1
python bin/iterate_davis2011kinase.py hybrid quad > iterate_davis2011kinase_hybrid_quad.log 2>&1

python bin/parse_log.py gp quad >> iterate_davis2011kinase_gp_quad.log 2>&1
python bin/parse_log.py mlper1 quad >> iterate_davis2011kinase_mlper1_quad.log 2>&1
python bin/parse_log.py mlper1g quad >> iterate_davis2011kinase_mlper1g_quad.log 2>&1
python bin/parse_log.py mlper5 quad >> iterate_davis2011kinase_mlper5_quad.log 2>&1
python bin/parse_log.py mlper5g quad >> iterate_davis2011kinase_mlper5g_quad.log 2>&1
python bin/parse_log.py hybrid quad >> iterate_davis2011kinase_hybrid_quad.log 2>&1

wait
