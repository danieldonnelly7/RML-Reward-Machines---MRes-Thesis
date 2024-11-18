### RML Reward Machines

This repository contains code for the experiments that were ran as part of my master's thesis. The thesis was based on RML Reward Machines, a novel reward machine variant that leverages RML, a specification language designed for runtime verification.

Individual experiments are contained in separate files. The naming convention starts with the environment used (letterenv or office world). To run the RML based experiments an RML monitor needs to be run locally with a port matching the relevant port for the environment (shown in the yaml files in the examples folder). More details on getting started with RML can be found at: https://rmlatdibris.github.io/

When a monitor with the correct specification is running, the experiments can be run individually br running the files. The results for the experiments, including code to plot graphs is contained within the results file.