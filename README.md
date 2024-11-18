### RML Reward Machines

This repository contains code for the experiments conducted as part of my master's thesis. The thesis focused on RML Reward Machines, a novel reward machine variant that leverages RML, a specification language designed for runtime verification.

#### **Overview**
- Individual experiments are contained in separate files.
- File names follow a convention starting with the environment used (`letterenv` or `office world`).
- The experiments include:
  - RML-based experiments.
  - Counting Reward Automata-based experiments.
  - Reward Machine-based experiments.
- Results, including code for plotting graphs, are available in the `results` folder.

#### **Running the Experiments**
1. **Set Up RML**:
   - To run RML-based experiments, you need an RML monitor running locally on a port matching the environment's port (specified in the YAML files in the `examples` folder).
   - More details on setting up RML can be found at [RML Documentation](https://rmlatdibris.github.io/).

2. **Run Experiments**:
   - Once the RML monitor is running, experiments can be executed by running the relevant files.

#### **Attribution**
- Code for RML-based experiments builds on the [RMLGym](https://github.com/hishamu7776/rml-gym).
- Code for Counting Reward Automata and Reward Machine-based experiments is adapted from [Counting Reward Automata](https://github.com/TristanBester/counting-reward-automata).

#### **Results**
- Results, including plotting code, are contained in the `results` folder.


