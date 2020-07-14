for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_gp.log
    python bin/gfp.py gp 1 $seed >> gfp_gp.log

    echo -e "GFP Seed:\t"$seed >> gfp_linear.log
    python bin/gfp.py linear 0 $seed >> gfp_linear.log

    echo -e "GFP Seed:\t"$seed >> gfp_hybrid.log
    python bin/gfp.py hybrid 1 $seed >> gfp_hybrid.log

    echo -e "GFP Seed:\t"$seed >> gfp_mlper5g.log
    python bin/gfp.py mlper5g 1 $seed >> gfp_mlper5g.log

    echo -e "GFP Seed:\t"$seed >> gfp_mlper1.log
    python bin/gfp.py mlper1 0 $seed >> gfp_mlper1.log
done

source activate edward
for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_bayesnn.log
    python bin/gfp.py bayesnn 1 $seed >> gfp_bayesnn.log
done
conda deactivate
