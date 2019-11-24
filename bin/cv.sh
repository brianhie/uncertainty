python bin/train_davis2011kinase.py gp >> train_davis2011kinase_gp.log 2>&1 &

for i in {0..4}
do
    export CUDA_VISIBLE_DEVICES=0
    python bin/train_davis2011kinase.py mlper1 >> train_davis2011kinase_mlper1.log 2>&1

    export CUDA_VISIBLE_DEVICES=0
    python bin/train_davis2011kinase.py dmlper1 >> train_davis2011kinase_dmlper1.log 2>&1 &

    export CUDA_VISIBLE_DEVICES=1
    python bin/train_davis2011kinase.py mlper5g >> train_davis2011kinase_mlper5g.log 2>&1 &

    export CUDA_VISIBLE_DEVICES=2
    python bin/train_davis2011kinase.py hybrid >> train_davis2011kinase_hybrid.log 2>&1 &

    export CUDA_VISIBLE_DEVICES=3
    python bin/train_davis2011kinase.py dhybrid >> train_davis2011kinase_dhybrid.log 2>&1 &

    wait
done

source activate edward
for i in {0..4}
do
    python bin/train_davis2011kinase.py bayesnn >> train_davis2011kinase_bayesnn.log 2>&1
done
conda deactivate

wait
