Ubuntu 18.04 for running `build_and_run_sim.sh`
=========================================================

- Pull the generic Ubuntu 18.04 image

  `docker pull ubuntu:18.04`

- Build a specific Ubuntu image to run our script

  `docker build --rm -t ubuntu-paraview-ascent .`

- Run the container ubuntu-paraview-ascent, using image ubuntu-paraview-ascent to
  create it, mounting the important folders. For instance ~/projects. This gives
  you a prompt in Ubuntu 18.04 docker.

  `docker run -it -v ~/projects/:/root/projects -v ~/tests-docker:/root/tests --name ubuntu-paraview-ascent ubuntu-paraview-ascent`

- Run test script.

  ```
  mkdir build
  cmake ..
  make
  ctest -D Experimental
  ```
  
  The additional parameter keep_going: optional count that says how
  many time we keep going when we should stop (3 means that we run and
  test the simulations)

- To add the script to crontab run:
  `crontab -e` and add the following line:

  `01 01  * * * cd /home/danlipsa/projects/ascent/src/examples/paraview-vis/tests/build && make clean && ctest -D Experimental`

  To run the `build_and_test.sh` script at 1:01 am.

- Exit the container

  `exit`

- Start the container after exit

  `docker container start ubuntu-paraview-ascent`

- If you need to, connect interactively to the container

  `docker exec -it ubuntu-paraview-ascent /bin/bash`
