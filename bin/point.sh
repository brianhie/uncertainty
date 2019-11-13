export CUDA_VISIBLE_DEVICES=2

# Point predictions only.

python bin/iterate_davis2011kinase.py gp pointexploit 10 > iterate_davis2011kinase_gp_pointexploit.log 2>&1 &

source activate edward
python bin/iterate_davis2011kinase.py bayesnn pointexploit 10 \
       > iterate_davis2011kinase_bayesnn_pointexploit.log 2>&1 &
conda deactivate

python bin/iterate_davis2011kinase.py mlper1 pointexploit 10 \
       > iterate_davis2011kinase_mlper1_pointexploit.log 2>&1
python bin/iterate_davis2011kinase.py dmlper1 pointexploit 10 \
       > iterate_davis2011kinase_dmlper1_pointexploit.log 2>&1
python bin/iterate_davis2011kinase.py mlper5g pointexploit 10 \
       > iterate_davis2011kinase_mlper5g_pointexploit.log 2>&1

wait

python bin/parse_log.py gp pointexploit >> iterate_davis2011kinase_gp_pointexploit.log 2>&1 &
python bin/parse_log.py bayesnn pointexploit >> iterate_davis2011kinase_bayesnn_pointexploit.log 2>&1 &
python bin/parse_log.py mlper1 pointexploit >> iterate_davis2011kinase_mlper1_pointexploit.log 2>&1 &
python bin/parse_log.py dmlper1 pointexploit >> iterate_davis2011kinase_dmlper1_pointexploit.log 2>&1 &
python bin/parse_log.py mlper5g pointexploit >> iterate_davis2011kinase_mlper5g_pointexploit.log 2>&1 &

wait
