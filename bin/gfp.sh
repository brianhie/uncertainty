for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_dhybrid.log
    python bin/gfp.py dhybrid 1 $seed >> gfp_dhybrid.log

    echo -e "GFP Seed:\t"$seed >> gfp_dmlper5g.log
    python bin/gfp.py dmlper5g 1 $seed >> gfp_dmlper5g.log

    echo -e "GFP Seed:\t"$seed >> gfp_dmlper1.log
    python bin/gfp.py dmlper1 0 $seed >> gfp_dmlper1.log
done

for seed in {1..5}
do
    echo -e "GFP Seed:\t"$seed >> gfp_bayesnn.log
    python bin/gfp.py bayesnn 1 $seed >> gfp_bayesnn.log
done
