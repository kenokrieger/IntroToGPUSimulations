# Simulating the Bornholdt Model

This repository contains example code for a guide on 
GPU accelerated simulations. The folders each contain
examples for the matching section name in the guide.

## 0. Getting started

To conduct your first simulation of the Bornholdt Model 
change into the directory ```0_Getting_Started``` and 
execute ```simulation.py```. This will record the relative
magnetisation during the simulation and save it as ```magnetisation.dat```

To examine your results, run ```validation.py```. This will produce
the two plots ```Magnetisation.pdf``` and 
```Cumulative_Return_Distribution.pdf```. Check your results. If you 
find an exponent of about 1 for the power law of the cumulative
return distribution, your findings match the expected results.

## 1. Preparing the GPU simulation

The first attempt at parallelisation runs exactly as the previous simulation,
except it now requires a configuration file instead of writing the 
parameters directly in the script. Again, change into
```1_Preparing_for_the_GPU``` and run ```simulation.py``` and 
```validation.py``` to generate some plots. Check your results. Notice also
that the simulation already runs about 10 times faster, but the code has
also got quite a bit longer.

## 2. The first GPU simulation

You may now try out to run the first GPU accelerated simulation.
First, you need to install the CUDA Programming Toolkit and Python's
numba module. Then you can change into the directory ```2_GPU_Simulation```
and execute ```simulation.py```. Compare the execution times to the
previous implementation. Why is it not faster as promised?

This is because the lattice we are simulating is too small for a GPU. Try 
increasing the lattice size and compare the spin updates per nanosecond.
You should notice that it drastically increases with lattice size. You
can go up to 20480 x 20480 lattices to see a performance increase.