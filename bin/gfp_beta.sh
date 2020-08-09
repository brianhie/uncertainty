betas=(0.0)

#for seed in {1..5}
#do
#    for b in ${betas[@]}
#    do
#        echo -e "GFP Seed:\t"$seed >> gfp_gp_exploit$b.log
#        python bin/gfp.py gp $b $seed >> gfp_gp_exploit$b.log
#
#        echo -e "GFP Seed:\t"$seed >> gfp_hybrid_exploit$b.log
#        python bin/gfp.py hybrid $b $seed >> gfp_hybrid_exploit$b.log
#
#        echo -e "GFP Seed:\t"$seed >> gfp_mlper5g_exploit$b.log
#        python bin/gfp.py mlper5g $b $seed >> gfp_mlper5g_exploit$b.log
#    done
#done

for seed in {1..5}
do
    for b in ${betas[@]}
    do
        echo -e "GFP Seed:\t"$seed >> gfp_bayesnn_exploit$b.log
        python bin/gfp.py bayesnn 1 $seed >> gfp_bayesnn_exploit$b.log
    done
done
