# RNN_tabletennis

LSTM ball trajectory prediction. Original idea from [Applying Deep Learning to Basketball Trajectories](https://arxiv.org/abs/1608.03793)

More detail in the jupyter notebook.

## Installation

1. Install tensorflow for running the python interface.

2. Following [tensorflow-cmake](https://github.com/cjweeks/tensorflow-cmake) to build tensorflow shared library, for running the c++ interface.

3. Train, or download our model.

Training data|Models
----|----
[coords.csv](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/blob/coords.csv)|[export-graph_125.pb](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/blob/export-graph_125.pb)
[coords_30.csv](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/blob/coords_30.csv)|[export-graph_30.pb](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/blob/export-graph_30.pb)

4. Build the C++ interface (Optional).
```bash
cd src
mkdir build && cd build
cmake ..
make
```
You have to modify some path in the `CMakeList.txt` file in order to build.

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

![Input Data](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/img/traj_data.png)

Trajectory prediction with 30 input data points

![Trajectory Prediction](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/img/traj_pred_30.png)

Trajectory prediction with **only** 4 input data points

![Trajectory Prediction 2](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/img/traj_pred_4.png)

![Training Loss](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/img/Screenshot_2017-03-11-12-11-46.png)

![Weights Evolution](https://brotherjing-static.s3-ap-northeast-1.amazonaws.com/img/Screenshot_2017-03-11-12-12-10.png)
