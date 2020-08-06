for i in {0..4}
do
    python bin/iterate_davis2011kinase.py gp quad 100 \
           --seed $i --beta 1 \
           > iterate_davis2011kinase_gp_quad.log 2>&1

    python bin/iterate_davis2011kinase.py cmf quad 100 --seed $i \
           >> iterate_davis2011kinase_cmf_quad.log 2>&1

    python bin/iterate_davis2011kinase.py mlper1 quad 100 --seed $i \
           >> iterate_davis2011kinase_mlper1_quad.log 2>&1

    python bin/iterate_davis2011kinase.py mlper5g quad 100 \
           --seed $i --beta 1 \
           >> iterate_davis2011kinase_mlper5g_quad.log 2>&1

    python bin/iterate_davis2011kinase.py hybrid quad 100 \
           --seed $i --beta 1 \
           >> iterate_davis2011kinase_hybrid_quad.log 2>&1
done

for i in {0..4}
do
    python bin/iterate_davis2011kinase.py bayesnn quad 100 \
           --seed $i --beta 1 \
           >> iterate_davis2011kinase_bayesnn_quad.log 2>&1
done
