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
that the simulation already runs about 10 times faster (woah!).

## 2. The first GPU simulation

