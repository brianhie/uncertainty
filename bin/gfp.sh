python bin/gfp.py gp 1 $seed >> gfp_gp.log
python bin/gfp.py linear 1 $seed >> gfp_linear.log

for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_hybrid.log
    python bin/gfp.py hybrid 1 $seed >> gfp_hybrid.log

    export CUDA_VISIBLE_DEVICES=1
    echo -e "GFP Seed:\t"$seed >> gfp_mlper5g.log
    python bin/gfp.py mlper5g 1 $seed >> gfp_mlper5g.log &

    export CUDA_VISIBLE_DEVICES=2
    echo -e "GFP Seed:\t"$seed >> gfp_mlper1.log
    python bin/gfp.py mlper1 0 $seed >> gfp_mlper1.log &

    wait
done

source activate edward
for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_bayesnn.log
    python bin/gfp.py bayesnn 1 $seed >> gfp_bayesnn.log
done
conda deactivate
