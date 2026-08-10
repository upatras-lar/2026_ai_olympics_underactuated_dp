## Real-Time Nonlinear MPC via Sequential Quadratic Programming for Underactuated Double-Pendulum Swing-Up

Submission to the [4th AI Olympics](https://ai-olympics.dfki-bremen.de/).

#### Team

- Nick Karydakis: [nickkarydakis@gmail.com](mailto:nickkarydakis@gmail.com)
- Konstantinos Chatzilygeroudis: [costashatz@upatras.gr](mailto:costashatz@upatras.gr)


### Usage instructions
To execute the Pendubot controller, follow these steps.

1. Upload these files to the Cloud Pendulum platform.
2. Run `setup.sh`. This will simply create folders for the output.
3. In the main directory, create a file named `token.txt`. Within, place the value of a valid user token, which has access to the Pendubot and Double Pendulum with valid experiment lengths (30 and 60). When running the notebooks, make sure a kernel that has access to `ssqpy` is used. This can be done by importing `ssqpy` from a notebook the main directory, which contains the `.so` file and then using that kernel for the rest of the notebooks, copying the `.so` file to each subdirectory, or correctly setting up a kernel that contains the main directory in its python path.

#### Notebooks
For each system, the `swingup` notebook runs the controller once on a random cell and plots the output data.

The `swingup_benchmark` notebook benchmarks the controller for a number of experiments (default 10) and creates output plots, as well as downloading the videos. On top of this, it saves a file of all the generated data.

The `swingup_disturbance_benchmark` notebook does the same, but applies torque disturbances to the system at certain times during the experiment, thus benchmarking the controller's ability to recover after a nominal push.

The `controller` notebook contains the same controller, using the required interface for the competition.


#### Overall plotting
After execution of both benchmarking notebooks finishes, two pickle files are saved with data from each. The `plot.py` script can then be executed to create an overall plot with the success rates and uptime of the benchmarks. The script contains a `SYSTEM` variable, which can be switched between `"pendubot"` and `"acrobot"`.