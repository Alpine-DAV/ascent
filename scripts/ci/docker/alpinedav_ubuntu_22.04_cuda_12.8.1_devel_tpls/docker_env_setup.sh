# create some helper scripts
python ascent/scripts/gen_spack_env_script.py cmake mpi
echo "git clone --recursive https://github.com/Alpine-DAV/ascent.git" > clone.sh
chmod +x clone.sh

# delete copy of source from host (ci will fetch new from repo)
rm -rf ascent