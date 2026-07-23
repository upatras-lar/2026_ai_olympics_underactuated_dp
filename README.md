## Real-Time Nonlinear MPC via Sequential Quadratic Programming for Underactuated Double-Pendulum Swing-Up

Submission to the [4th AI Olympics](https://ai-olympics.dfki-bremen.de/).

#### Team

- Nick Karydakis: [nickkarydakis@gmail.com](mailto:nickkarydakis@gmail.com)
- Konstantinos Chatzilygeroudis: [costashatz@upatras.gr](mailto:costashatz@upatras.gr)


### Usage instructions
To execute the Pendubot controller, follow these steps.

1. Upload these files to the Cloud Pendulum platform.
2. Run `setup.sh`. This will simply create folders for the output.
3. Each notebook contains a `user_token` variable. Replace the value of this with a valid user token, which has access to the Pendubot with valid experiment lengths (30 and 60).

#### Notebooks
The `swingup` notebook runs the controller once on a random cell and plots the output data.

The `swingup_benchmark` notebook benchmarks the controller for a number of experiments (default 10) and creates output plots, as well as downloading the videos. On top of this, it saves a file of all the generated data.

The `swingup_disturbance_benchmark` notebook does the same, but applies torque disturbances to the Pendubot at certain times during the experiment, thus benchmarking the controller's ability to recover after a nominal push.


#### Overall plotting
After execution of both benchmarking notebooks finishes, two pickle files are saved with data from each. The `plot.py` script can then be executed to create an overall plot with the success rates and uptime of the benchmarks.