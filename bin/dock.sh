docking_dir=data/docking

input_dirname=$docking_dir/structure_files
target_fname=$input_dirname/2FUM_chainA_nohet.pdbqt
conf_fname=$input_dirname/2FUM_chainA_nohet_conf.txt

out_dirname=$docking_dir/docked_files
log_dirname=$docking_dir/log_files

ls $input_dirname | \
    grep 'design_gp' | \
    grep '\.pdbqt' | \
    grep -v 2FUM | \
    while read ligand_fname
    do
        echo -e "\n${ligand_fname}\n"
        vina \
            --receptor $target_fname \
            --ligand $input_dirname/$ligand_fname \
            --config $conf_fname \
            --out $out_dirname/$ligand_fname.out.pdbqt \
            --log $log_dirname/$ligand_fname.log
    done
