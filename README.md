# Simulating the Bornholdt Model

This repository contains example code for a guide on 
GPU accelerated simulations. The folders each contain
examples for the matching section name in the guide.

**I wrote all the programs using Jetbrain's respective IDEs
(free pro version for students) on my Ubuntu 20.04 machine.
Most if not all the code should be cross-platform compatible,
but I did not test it on any other OS. If you do, feel free to
share your experiences with me!**

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

## 3. Moving to C

The simulation now runs reasonably fast, but we can do better. However,
we need to venture deeper in the programming world and use a lower
level language such as C. C has the extension CUDA C which allows us to
take full control of our GPU. To prepare for this, we first take a step
back and write a simple C/C++ program as a little warm up.

To run the program change into the directory ```3_Moving_to_C``` and 
build the application using CMake (if you are new to working with
compiling and building a program, I recommend opening the directory with
Jetbrain's IDE Clion as it will take care of all the compiling for you).
You can then run the application either in the IDE of your choice or 
in the command line with ```./multising```. We lost some computation 
speed but drastically outperformed the raw python implementation.
Again, you can check your results by executing ```validation.py```.

## 4. The second GPU simulation

With our new gained knowledge, it is now time to revisit our GPU.

To run the program change into ```4_CUDA_C``` and build the application
using CMake. This time it requires to add the ```$CUDA_INSTALLATION/include```
as a path for includes to the compiler. You can run the program just as 
before with one minor change: the parameters ```grid_height``` and
```grid_width``` are now called ```lattice_height``` and
```lattice_width``` to not confuse them with the name ```grid```
from the NVIDIA programming model. You can check the computation speed
and see the same phenomenon as with the Python GPU implementation. The 
simulation is also roughly two times faster than its counterpart in Python
but also roughly 10 times more complex to read.

## 5. Give it all we got

