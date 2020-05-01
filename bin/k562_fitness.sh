echo -e "K562FIT Seed:\t"$seed >> k562fit_gp.log
python bin/k562_fitness.py gp exploit 100 $seed >> k562fit_gp.log &

for seed in {1..5}
do
    export CUDA_VISIBLE_DEVICES=1
    echo -e "K562FIT Seed:\t"$seed >> k562fit_hybrid.log
    python bin/k562_fitness.py hybrid exploit 100 $seed >> k562fit_hybrid.log &

    export CUDA_VISIBLE_DEVICES=2
    echo -e "K562FIT Seed:\t"$seed >> k562fit_mlper5g.log
    python bin/k562_fitness.py mlper5g exploit 100 $seed >> k562fit_mlper5g.log &

    wait

    export CUDA_VISIBLE_DEVICES=1
    echo -e "K562FIT Seed:\t"$seed >> k562fit_mlper1.log
    python bin/k562_fitness.py mlper1 exploit 100 $seed >> k562fit_mlper1.log &

    export CUDA_VISIBLE_DEVICES=2
    echo -e "K562FIT Seed:\t"$seed >> k562fit_cmf.log
    python bin/k562_fitness.py cmf exploit 100 $seed >> k562fit_cmf.log &

    wait
done

#for seed in {1..5}
#do
#    echo -e "K562FIT Seed:\t"$seed >> k562fit_bayesnn.log
#    python bin/k562_fitness.py bayesnn exploit 100 $seed >> k562fit_bayesnn.log
#done
