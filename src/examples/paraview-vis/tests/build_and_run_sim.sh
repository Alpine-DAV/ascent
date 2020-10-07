#!/bin/bash

# Remove older spack package described by 'spec'
# We assume only 2 packages are installed which is
# the case for a spack repo only updated with this script
# Parameters:
# spec: spack spec for which to remove the older version
removeOlder()
{
    if [ $# -ne 1 ]; then
        echo "${FUNCNAME[0]} spec"
        return 1
    fi
    local spec=$1
    local count=0
    local sha
    local newestSha
    local dir
    while IFS=' ' read -r sha _ dir
    do
        # there are two extra lines installed by
        if (( count == 2 )); then
            local date0
            date0="$(stat -c %Y "$dir")"
            local sha0=$sha
            newestSha=$sha0
        elif (( count == 3 )); then
            local date1
            date1="$(stat -c %Y "$dir")"
            local sha1=$sha
            if (( date0 < date1 )); then
                echo "spack uninstall -y --dependents /$sha0"
                spack uninstall -y --dependents "/$sha0"
                newestSha=$sha1
            else
                echo "spack uninstall -y --dependents /$sha1"
                spack uninstall -y --dependents "/$sha1"
                newestSha=$sha0
            fi
        fi
        count=$((count+1))
    done < <(spack find -lp "$spec")
    echo "$spec: $newestSha"
}

# Setup, run and test simulation
# Needs ImageMagic (compare) and bc (for float comparison)
# Parameters:
# dirsim: directory and simulation to run
testSimulation()
{
    if [ $# -ne 1 ]; then
        echo "${FUNCNAME[0]} dirsim"
        return 1
    fi
    local dirsim=$1
    local dir=${dirsim%/*}
    local sim=${dirsim##*/}

    echo "Running $sim ..."
    mkdir -p "$sim"
    cd "$sim" || return 1
    rm -f ./*
    ln -s "$(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py"
    ln -s "$(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview-vis-$sim.py" paraview-vis.py
    ln -s "$(spack location --install-dir ascent)/examples/ascent/paraview-vis/ascent_actions.json"
    if [ "$sim" = "cloverleaf3d" ]; then
        ln -s "$(spack location --install-dir ascent)/examples/ascent/$dir/$sim/clover.in"
        "$(spack location --install-dir mpi)/bin/mpiexec" -n 2 "$(spack location --install-dir ascent)/examples/ascent/$dir/$sim/${sim}_par" > output.txt 2>&1
    elif [ "$sim" = "kripke" ]; then
        "$(spack location --install-dir mpi)/bin/mpiexec" -np 8 "$(spack location --install-dir ascent)/examples/ascent/$dir/$sim/${sim}_par" --procs 2,2,2 --zones 32,32,32 --niter 5 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4 > output.txt 2>&1
    elif [ "$sim" = "lulesh" ]; then
        "$(spack location --install-dir mpi)/bin/mpiexec" -np 8 "$(spack location --install-dir ascent)/examples/ascent/$dir/$sim/${sim}_par" -i 10 -s 32 > output.txt 2>&1
    elif [ "$sim" = "noise" ]; then
        "$(spack location --install-dir mpi)/bin/mpiexec" -np 8 "$(spack location --install-dir ascent)/examples/ascent/$dir/$sim/${sim}_par"  --dims=32,32,32 --time_steps=5 --time_delta=1 > output.txt 2>&1
    fi
    local file
    for file in image_*.png; do
        local base=${file%%.*}
        # normalized cross corelation
        local ncc
        # don't test the return value as it returns 1 (error) if not exact comparison
        ncc=$(compare -metric NCC "$file" "$(spack location --install-dir ascent)/examples/ascent/paraview-vis/tests/baseline_images/$sim/$file" "diff_${base}.png" 2>&1)
        local lessThan
        if ! lessThan=$(echo "${ncc} < 0.98" | bc -l); then
            echo "Error: $file too different from baseline"
            return 1
        fi
        if [[ "$lessThan" == 1 ]]; then
            echo "Error $lessThan for $sim: $file <> $(spack location --install-dir ascent)/examples/ascent/paraview-vis/tests/baseline_images/$sim/$file"
            cd ..
            return 1
        else
            echo "$sim: $file OK"
        fi
    done
    echo "Passed tests for $sim"
    cd ..
}

# Extract name, version and if its static from spec
# Parameter:
# spec: spack spec
dirFromSpec()
{
    if [[ $# -ne 1 ]]; then
        echo "${FUNCNAME[0]} spec"
        return 1
    fi
    spec=$1
    name=${spec%%@*}
    rest=${spec#*@}
    version=${rest%%[+~]*}
    dir="${name}_${version}"
    if [[ "$rest" == *"~shared"* ]]; then
        dir="${dir}_static"
    fi
    echo "$dir"
}

# Install ParaView and ascent spack packages, remove duplicate packages,
# create simulation directories, run simulations and check results
# Parameters:
# paraview_spec, ascent_spec: spack specs for paraview and ascent to install
# Dynamic parameters (see main):
# spack_dir, sim_dir, build_option and keep_going
testParaViewAscent()
{
    if [[ $# -ne 2 ]]; then
        echo "${FUNCNAME[0]} paraview_spec ascent_spec"
        return 1
    fi
    paraviewSpec=$1
    ascentSpec=$2
    echo "Testing $paraviewSpec $ascentSpec ..."
    # build paraview and ascent
    # shellcheck source=/home/danlipsa/projects/spack/share/spack/setup-env.sh
    . ${spack_dir}/share/spack/setup-env.sh
    spack install "$build_option" "$paraviewSpec" | tee result.txt;
    if [[ ${PIPESTATUS[0]} -gt 0 ]]; then
        return 1
    fi
    local installed=0
    if grep "Installing paraview" < result.txt; then
        installed=$((installed + 1))
    fi
    spack install "$build_option" "$ascentSpec" | tee result.txt
    if [[ ${PIPESTATUS[0]} -gt 0 ]]; then
        return 1
    fi
    if grep "Installing ascent" < result.txt; then
        installed=$((installed + 1))
    fi
    rm -f result.txt
    if [[ $installed -eq 0 && "$keep_going" -eq 0 ]]; then
        echo "No new package installed, no need to run tests."
    else
        echo "Remove older packages ..."
        local paraviewSha
        paraviewSha=$(removeOlder "$paraviewSpec")
        echo "$paraviewSha"
        paraviewSha=${paraviewSha#*: }
        removeOlder mpich
        removeOlder py-mpi4py
        removeOlder py-numpy
        removeOlder python
        removeOlder conduit
        # load packages
        spack load conduit;spack load python;spack load py-numpy;spack load py-mpi4py;spack load "paraview/$paraviewSha"
        # $sim_dir has dynamic scope
        cd "$sim_dir" || return 1
        paraviewDirName=$(dirFromSpec "$paraviewSpec")
        ascentDirName=$(dirFromSpec "$ascentSpec")
        mkdir -p "${paraviewDirName}_${ascentDirName}"
        cd "${paraviewDirName}_${ascentDirName}" || return 1
        testsPassed=0
        if testSimulation "proxies/cloverleaf3d"; then
            testsPassed=$((testsPassed + 1))
        fi
        if testSimulation "proxies/kripke"; then
            testsPassed=$((testsPassed + 1))
        fi
        if testSimulation "proxies/lulesh"; then
            testsPassed=$((testsPassed + 1))
        fi
        if testSimulation "synthetic/noise"; then
            testsPassed=$((testsPassed + 1))
        fi
        echo "Tests passed: $testsPassed of 4"        
        if [[ $testsPassed -ne 4 ]]; then
            return 1
        fi
    fi
}


# Install several versions of ParaView and ascent spack packages, remove duplicate packages,
# create simulation directories, run simulations and check results
# Parameters:
# spack_dir: directory where spack repo is cloned
# sim_dir: directory where to run simulations
# keep_going: optional count that says how many time we keep going when we should stop
# build_option: option to build packages such as -j40
# build_dependency: dependency to fix spack issues (such as ^mpich)
main()
{
    if [ $# -lt 2 ]; then
        echo "$0 spack_dir sim_dir [keep_going] [build_option] [build_dependency]"
        return 1
    fi
    local spack_dir=$1
    local sim_dir=$2
    local keep_going=$3
    if [[ -z $keep_going ]]; then
        keep_going=0
    fi
    local build_option=$4
    if [[ -z $build_option ]]; then
        build_option=""
    fi
    local build_dependency=$5
    if [[ -z $build_dependency ]]; then
        build_dependency=""
    fi
    cd "$spack_dir" || return 1
    # update spack
    local result
    if ! result=$(git pull) && [[ "$keep_going" -eq 0 ]]; then
        return 1
    fi
    if [[ $keep_going -gt 0 ]]; then
        keep_going=$((keep_going - 1))
    fi
    echo "$result"
    if [[ "$result" = "Already up to date." && "$keep_going" -eq 0 ]]; then
        echo "Repository up to date, no need to run tests"
        return 0
    fi
    if [[ $keep_going -gt 0 ]]; then
        keep_going=$((keep_going - 1))
    fi
    if ! testParaViewAscent paraview@develop+python3+mpi+osmesa+shared${build_dependency} ascent@develop~vtkh${build_dependency}; then
        return 1
    fi
    if ! testParaViewAscent paraview@develop+python3+mpi+osmesa~shared${build_dependency} ascent@develop~vtkh${build_dependency}; then
        return 1
    fi
    echo "All Ascent ParaView tests passed"
    return 0
}

main "$@"
