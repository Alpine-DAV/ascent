# Kripke with Ascent Integration

Sample Run:

mpiexec -n 1 ./kripke_steering_par --procs 1,1,1 --zones 32,32,32 --niter 3 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4

or

mpiexec -n 8 ./kripke_steering_par --procs 2,2,2 --zones 32,32,32 --niter 5 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4

For more info, please visit:

http://ascent.readthedocs.io/en/latest/ExampleIntegrations.html
