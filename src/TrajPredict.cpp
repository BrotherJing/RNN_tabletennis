#include "TrajPredict.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flags;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

const char* placeholder_x = "Test/Model/Input_data:0";
const char* initial_state_c_0 = "Test/Model/zeros:0";
const char* initial_state_h_0 = "Test/Model/zeros_1:0";
const char* initial_state_c_1 = "Test/Model/zeros_2:0";
const char* initial_state_h_1 = "Test/Model/zeros_3:0";

const char* final_state_c_0 = "Test/Model/RNN/multi_rnn_cell/cell_0/lstm_cell/add_3:0";
const char* final_state_h_0 = "Test/Model/RNN/multi_rnn_cell/cell_0/lstm_cell/mul_5:0";
const char* final_state_c_1 = "Test/Model/RNN/multi_rnn_cell/cell_1/lstm_cell/add_3:0";
const char* final_state_h_1 = "Test/Model/RNN/multi_rnn_cell/cell_1/lstm_cell/mul_5:0";

const char* mu1 = "Test/Model/MDN/split_1:0";
const char* mu2 = "Test/Model/MDN/split_1:1";
const char* mu3 = "Test/Model/MDN/split_1:2";
const char* s1 = "Test/Model/MDN/Exp_1:0";
const char* s2 = "Test/Model/MDN/Exp_2:0";
const char* s3 = "Test/Model/MDN/Exp_3:0";
const char* rho = "Test/Model/MDN/Tanh:0";
const char* theta = "Test/Model/MDN/Mul:0";

const int BATCH_SIZE = 20;
const int SEQUENCE_LENGTH = 120;
const int NUM_COORDS = 3;
const int SEQ_PRE = 30;
const int PREDICT_LEN = 100;

const float X_NORM = 1525;
const float Y_NORM = 2740;
const float Z_NORM = 458;

Status TrajPredict::initialize(std::unique_ptr<Session> *session, std::vector<Tensor> *initialized_outputs){
  Status run_initialize_status = (*session)->Run({}, {
    initial_state_c_0,
    initial_state_h_0,
    initial_state_c_1,
    initial_state_h_1
  }, {}, initialized_outputs);
  return run_initialize_status;
}

int TrajPredict::sampleTheta(Tensor &theta, int idx){
  float stop = rand()*1.0/RAND_MAX;
  float cum = 0.;
  int num_thetas = theta.dim_size(1);
  auto tensor = theta.tensor<float, 3>();
  for(int i=0;i<num_thetas;++i){
    cum += tensor(0, i, idx);
    if(cum>stop)return i;
  }
  return 0;
}

Status TrajPredict::sample(std::vector<Coords> &seq,
  std::vector<Coords> &seq_pred,
  int predict_len, 
  int sl_pre){

  double bias = 1.0;
  double mean[3];
  double stderr[4];

  std::vector<Coords> seq_feed(predict_len+1);
  for(int i=0;i<sl_pre+1;++i){
    seq_feed[i] = seq[i];
  }

  std::vector<Tensor> outputs;

  for(int i=0;i<predict_len;++i){

    fillPlaceholder(seq_feed, inputs[4].second, i);
  
    Status run_sample = (session)->Run(inputs, {
      mu1, mu2, mu3,
      s1, s2, s3,
      rho, theta,
      final_state_c_0, final_state_h_0,
      final_state_c_1, final_state_h_1
    }, {}, &outputs);
    
    if(!run_sample.ok()){
      return run_sample;
    }
    if(!(inputs[0].second.CopyFrom(outputs[8], outputs[8].shape())&&
    inputs[1].second.CopyFrom(outputs[9], outputs[9].shape())&&
    inputs[2].second.CopyFrom(outputs[10], outputs[10].shape())&&
    inputs[3].second.CopyFrom(outputs[11], outputs[11].shape()))){
      return Status::OK();
    }
    int idx = sampleTheta(outputs[7], 0);//theta
    mean[0] = outputs[0].tensor<float, 3>()(0, idx, 0);
    mean[1] = outputs[1].tensor<float, 3>()(0, idx, 0);
    mean[2] = outputs[2].tensor<float, 3>()(0, idx, 0);
    stderr[0] = exp(-1*bias)*outputs[3].tensor<float, 3>()(0, idx, 0);
    stderr[1] = exp(-1*bias)*outputs[4].tensor<float, 3>()(0, idx, 0);
    stderr[2] = exp(-1*bias)*outputs[5].tensor<float, 3>()(0, idx, 0);
    stderr[3] = outputs[6].tensor<float, 3>()(0, idx, 0)*stderr[0]*stderr[1];
    
    Eigen::MatrixXd covar(3,3);
    covar << stderr[0]*stderr[0], stderr[3], 0,
          stderr[3], stderr[1]*stderr[1], 0,
          0, 0, stderr[2]*stderr[2];
    Eigen::VectorXd means(3);
    means << mean[0], mean[1], mean[2];

    normal_random_variable sample { means, covar };
    
    Eigen::VectorXd draw = sample();
    if(i>=sl_pre)
      seq_feed[i+1] = Coords{seq_feed[i].x+float(draw(0)), seq_feed[i].y+float(draw(1)), seq_feed[i].z+float(draw(2))};
  }
  for(int i=0;i<predict_len;++i){
    seq_pred.push_back(Coords{seq_feed[i].x*X_NORM, seq_feed[i].y*Y_NORM, seq_feed[i].z*Z_NORM});
  }
  return Status::OK();
}

Status TrajPredict::LoadGraph(const string graph_file_name, std::unique_ptr<Session> *session){
  GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
  if(!load_graph_status.ok()){
    return errors::NotFound("Failed to load graph at '", graph_file_name, "'");
  }
  session->reset(NewSession(SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if(!session_create_status.ok()){
    return session_create_status;
  }
  return Status::OK();
}

void TrajPredict::fillPlaceholder(std::vector<Coords> &seq, Tensor &x, int idx){
  auto tensor = x.tensor<float, 3>();
  Coords coord = seq[idx];
  tensor(0,0,0) = coord.x;
  tensor(0,1,0) = coord.y;
  tensor(0,2,0) = coord.z;
}

TrajPredict::TrajPredict(const string graph_file_name){
  srand(time(NULL));
  Status status = LoadGraph(graph_file_name, &session);
  if (!status.ok()) {
    LOG(ERROR)<<status;
    return;
  }
  Status run_initialize_status = initialize(&session, &initialized_outputs);
  if(!run_initialize_status.ok()){
    LOG(ERROR)<<run_initialize_status;
    return;
  }
  Tensor x(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_COORDS, 1}));
  inputs = {
    {initial_state_c_0, initialized_outputs[0]},
    {initial_state_h_0, initialized_outputs[1]},
    {initial_state_c_1, initialized_outputs[2]},
    {initial_state_h_1, initialized_outputs[3]},
    {placeholder_x, x}
  };
}

TrajPredict::~TrajPredict(){
  session->Close();
}
