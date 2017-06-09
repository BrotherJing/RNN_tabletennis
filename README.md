# RNN_tabletennis

LSTM ball trajectory prediction. Original idea from [Applying Deep Learning to Basketball Trajectories](https://arxiv.org/abs/1608.03793)

More detail in the jupyter notebook.

## Installation

1. Install tensorflow for running the python interface.

2. Following [tensorflow-cmake](https://github.com/cjweeks/tensorflow-cmake) to build tensorflow shared library, for running the c++ interface.

3. Train, or download our model.

Training data|Models
----|----
[coords.csv](http://or9ajn83e.bkt.clouddn.com/coords.csv)|[export-graph_125.pb](http://or9ajn83e.bkt.clouddn.com/export-graph_125.pb)
[coords_30.csv](http://or9ajn83e.bkt.clouddn.com/coords_30.csv)|[export-graph_30.pb](http://or9ajn83e.bkt.clouddn.com/export-graph_30.pb)

4. Build the C++ interface (Optional).
```bash
cd src
mkdir build && cd build
cmake ..
make
```

## Usage

Train the model:
```bash
python main.py
```
Convert to .pb format:
```bash
python write_pb.py
```
Test the model:
```bash
python test_on_pb.py #or use the jupyter notebook
```

## Visualization

Input data

![Input Data](http://7xrcar.com1.z0.glb.clouddn.com/traj_data.png)

Trajectory prediction with 30 input data points

![Trajectory Prediction](http://7xrcar.com1.z0.glb.clouddn.com/traj_pred_30.png)

Trajectory prediction with **only** 4 input data points

![Trajectory Prediction 2](http://7xrcar.com1.z0.glb.clouddn.com/traj_pred_4.png)

![Training Loss](http://7xrcar.com1.z0.glb.clouddn.com/Screenshot%20from%202017-03-11%2012-11-46.png)

![Weights Evolution](http://7xrcar.com1.z0.glb.clouddn.com/Screenshot%20from%202017-03-11%2012-12-10.png)
