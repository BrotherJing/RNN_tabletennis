# RNN_tabletennis

LSTM ball trajectory prediction. Original idea from [Applying Deep Learning to Basketball Trajectories](https://arxiv.org/abs/1608.03793)

More detail in the jupyter notebook.

## Installation

1. Install tensorflow for running the python interface.

2. Following [tensorflow-cmake](https://github.com/cjweeks/tensorflow-cmake) to build tensorflow shared library, for running the c++ interface.

3. Download our model from...(TODO)

Visualize the input data

![Input Data](http://7xrcar.com1.z0.glb.clouddn.com/traj_data.png)

Trajectory prediction with 30 input data points

![Trajectory Prediction](http://7xrcar.com1.z0.glb.clouddn.com/traj_pred_30.png)

Trajectory prediction with **only** 4 input data points

![Trajectory Prediction 2](http://7xrcar.com1.z0.glb.clouddn.com/traj_pred_4.png)

![Training Loss](http://7xrcar.com1.z0.glb.clouddn.com/Screenshot%20from%202017-03-11%2012-11-46.png)

![Weights Evolution](http://7xrcar.com1.z0.glb.clouddn.com/Screenshot%20from%202017-03-11%2012-12-10.png)
