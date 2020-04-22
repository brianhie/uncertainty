docking_dir=data/docking

input_dirname=$docking_dir/structure_files
target_fname=$input_dirname/2FUM_chainA_nohet.pdbqt
conf_fname=$input_dirname/2FUM_chainA_nohet_conf.txt

out_dirname=$docking_dir/docked_files
log_dirname=$docking_dir/log_files

ledock $input_dirname/2FUM_chainA_nohet_conf_ledock.txt &

rdock_dir=$docking_dir/rdock_files
export RBT_ROOT=$rdock_dir

rbcavity \
    -r $input_dirname/2FUM_chainA_nohet_conf.prm \
    -was
ls $input_dirname | \
    grep 'real_' | \
    grep '\.sd' | \
    grep -v 2FUM | \
    while read ligand_fname
    do
        echo -e "\nrdock of ${ligand_fname}\n"
        rbdock \
            -i $input_dirname/$ligand_fname \
            -o $out_dirname/$ligand_fname.rdock.out \
            -r $input_dirname/2FUM_chainA_nohet_conf.prm \
            -p $rdock_dir/data/scripts/dock.prm \
            -n 50
    done

ls $input_dirname | \
    grep 'real_' | \
    grep '\.pdbqt' | \
    grep -v 2FUM | \
    while read ligand_fname
    do
        echo -e "\n${ligand_fname}\n"

        smina \
            --receptor $target_fname \
            --ligand $input_dirname/$ligand_fname \
            --config $conf_fname \
            --scoring vinardo \
            --out $out_dirname/$ligand_fname.vinardo.out.pdbqt \
            --log $log_dirname/$ligand_fname.vinardo.log \
            --cpu 48

        smina \
            --receptor $target_fname \
            --ligand $input_dirname/$ligand_fname \
            --config $conf_fname \
            --scoring dkoes_scoring  \
            --out $out_dirname/$ligand_fname.dk.out.pdbqt \
            --log $log_dirname/$ligand_fname.dk.log \
            --cpu 48

        smina \
            --receptor $target_fname \
            --ligand $input_dirname/$ligand_fname \
            --config $conf_fname \
            --out $out_dirname/$ligand_fname.smina.out.pdbqt \
            --log $log_dirname/$ligand_fname.smina.log \
            --cpu 48

        vina \
            --receptor $target_fname \
            --ligand $input_dirname/$ligand_fname \
            --config $conf_fname \
            --out $out_dirname/$ligand_fname.vina.out.pdbqt \
            --log $log_dirname/$ligand_fname.vina.log \
            --cpu 48
    done

wait
