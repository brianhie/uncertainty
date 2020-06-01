for seed in {0..4}
do
    echo -e "Norman Seed:\t"$seed >> norman_gp.log
    python bin/dataset_norman2019_k562.py perfect gp 1 --n-acquire 1000 --seed $seed >> norman_gp.log 2>&1 &

    echo -e "Norman Seed:\t"$seed >> norman_hybrid.log
    python bin/dataset_norman2019_k562.py perfect hybrid 1 --n-acquire 1000 --seed $seed >> norman_hybrid.log 2>&1 &

    echo -e "Norman Seed:\t"$seed >> norman_mlper1.log
    python bin/dataset_norman2019_k562.py perfect mlper1 1 --n-acquire 1000 --seed $seed >> norman_mlper1.log 2>&1

    echo -e "Norman Seed:\t"$seed >> norman_mlper5g.log
    python bin/dataset_norman2019_k562.py perfect mlper5g 1 --n-acquire 1000 --seed $seed >> norman_mlper5g.log 2>&1
done

source activate edward
for seed in {0..4}
do
    echo -e "Norman Seed:\t"$seed >> norman_bayesnn.log
    python bin/dataset_norman2019_k562.py perfect bayesnn 1 --n-acquire 1000 --seed $seed >> norman_bayesnn.log 2>&1
done
conda deactivate
