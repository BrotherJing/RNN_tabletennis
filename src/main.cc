#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include <stdlib.h>//srand, rand
#include <time.h>//time

#include <Eigen/Eigenvalues> 

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flags;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

#include "main.h"

Status initialize(std::unique_ptr<Session> *session, std::vector<Tensor> *initialized_outputs){
  Status run_initialize_status = (*session)->Run({}, {
    initial_state_c_0,
    initial_state_h_0,
    initial_state_c_1,
    initial_state_h_1
  }, {}, initialized_outputs);
  return run_initialize_status;
}

int sampleTheta(Tensor &theta, int idx){
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

Status sample(std::unique_ptr<Session> *session, 
  std::vector<Coords> &seq,
  std::vector<Coords> &seq_pred,
  std::vector<std::pair<tensorflow::string, Tensor> > &inputs,
  int predict_len, 
  int sl_pre){

  double bias = 1.0;
  double mean[3];
  double stderr[4];

  std::vector<Coords> seq_feed((predict_len/SEQUENCE_LENGTH+1)*SEQUENCE_LENGTH);
  for(int i=0;i<sl_pre+1;++i){
    seq_feed[i] = seq[i];
  }

  std::vector<Tensor> outputs;

  for(int i=sl_pre;i<predict_len;++i){
    int sl_draw = i % SEQUENCE_LENGTH;
    int block = i / SEQUENCE_LENGTH;
    fillPlaceholder(seq_feed, inputs[4].second, block*SEQUENCE_LENGTH, SEQUENCE_LENGTH);

    Status run_sample = (*session)->Run(inputs, {
      final_state_c_0, final_state_h_0,
      final_state_c_1, final_state_h_1,
      mu1, mu2, mu3,
      s1, s2, s3,
      rho, theta
    }, {}, &outputs);
    if(!run_sample.ok()){
      return run_sample;
    }
    if(sl_draw == SEQUENCE_LENGTH-1){
      if(!(inputs[0].second.CopyFrom(outputs[0], outputs[0].shape())&&
      inputs[1].second.CopyFrom(outputs[1], outputs[1].shape())&&
      inputs[2].second.CopyFrom(outputs[2], outputs[2].shape())&&
      inputs[3].second.CopyFrom(outputs[3], outputs[3].shape()))){
        return Status::OK();
      }
    }
    int idx = sampleTheta(outputs[11], sl_draw);//theta
    mean[0] = outputs[4].tensor<float, 3>()(0, idx, sl_draw);
    mean[1] = outputs[5].tensor<float, 3>()(0, idx, sl_draw);
    mean[2] = outputs[6].tensor<float, 3>()(0, idx, sl_draw);
    stderr[0] = exp(-1*bias)*outputs[7].tensor<float, 3>()(0, idx, sl_draw);
    stderr[1] = exp(-1*bias)*outputs[8].tensor<float, 3>()(0, idx, sl_draw);
    stderr[2] = exp(-1*bias)*outputs[9].tensor<float, 3>()(0, idx, sl_draw);
    stderr[3] = outputs[10].tensor<float, 3>()(0, idx, sl_draw)*stderr[0]*stderr[1];
    
    Eigen::MatrixXd covar(3,3);
    covar << stderr[0]*stderr[0], stderr[3], 0,
          stderr[3], stderr[1]*stderr[1], 0,
          0, 0, stderr[2]*stderr[2];
    Eigen::VectorXd means(3);
    means << mean[0], mean[1], mean[2];

    normal_random_variable sample { means, covar };
    
    Eigen::VectorXd draw = sample();
    seq_feed[i+1] = Coords{seq_feed[i].x+float(draw(0)), seq_feed[i].y+float(draw(1)), seq_feed[i].z+float(draw(2))};
  }
  for(int i=0;i<predict_len;++i){
    seq_pred.push_back(Coords{seq_feed[i].x*X_NORM, seq_feed[i].y*Y_NORM, seq_feed[i].z*Z_NORM});
  }
  return Status::OK();
}

Status LoadGraph(string graph_file_name, std::unique_ptr<Session> *session){
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

void loadSequence(string input_seq, std::vector<Coords> &seq){
  float x,y,z,max_z=0;
  std::ifstream input;
  input.open(input_seq.c_str(), std::ifstream::in);
  while(!input.eof()){
    input>>x>>y>>z;
    Coords coord{x/X_NORM,y/Y_NORM,z/Z_NORM};
    seq.push_back(coord);
  }
  input.close();
}

void saveSequence(string output_seq, std::vector<Coords> &seq){
  std::ofstream output;
  output.open(output_seq.c_str(), std::ios::out|std::ios::trunc);
  for(int i=0;i<seq.size();++i){
    output<<seq[i].x<<" "<<seq[i].y<<" "<<seq[i].z<<std::endl;
  }
  output.close();
}

void fillPlaceholder(std::vector<Coords> &seq, Tensor &x, int offset, int length){
  auto tensor = x.tensor<float, 3>();
  for(int i=0;i<length;++i){
    Coords coord = seq[i+offset];
    tensor(0,0,i) = coord.x;
    tensor(0,1,i) = coord.y;
    tensor(0,2,i) = coord.z;
  }
}

int main(int argc, char **argv) {

  srand(time(NULL));

  string graph = "/home/jing/Documents/RNN_tabletennis/data/export-graph.pb";
  string input_seq = "input.csv";
  string output_seq = "output.csv";

  std::vector<Flag> flag_list = {
    Flag("graph", &graph, "graph to be excuted"),
    Flag("input", &input_seq, "input sequence file"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if(!parse_result){
    LOG(ERROR) << usage;
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if(argc > 1){
    LOG(ERROR)<<"Unknown argument "<<argv[1]<<"\n"<<usage;
    return -1;
  }

  std::unique_ptr<Session> session;
  Status status = LoadGraph(graph, &session);
  if (!status.ok()) {
    LOG(ERROR)<<status;
    return -1;
  }
  std::cout<<"model loaded"<<std::endl;

  std::vector<Tensor> initialized_outputs;
  Status run_initialize_status = initialize(&session, &initialized_outputs);
  if(!run_initialize_status.ok()){
    LOG(ERROR)<<run_initialize_status;
    return -1;
  }
  std::cout<<"initialized"<<std::endl;

  std::vector<Coords> seq;
  std::vector<Coords> seq_pred;
  loadSequence(input_seq, seq);

  Tensor x(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_COORDS, SEQUENCE_LENGTH}));

  std::vector<std::pair<tensorflow::string, Tensor> > inputs = {
    {initial_state_c_0, initialized_outputs[0]},
    {initial_state_h_0, initialized_outputs[1]},
    {initial_state_c_1, initialized_outputs[2]},
    {initial_state_h_1, initialized_outputs[3]},
    {placeholder_x, x}
  };
  for(int i=0;i<x.dims();++i){
    std::cout<<"dims "<<i<<": "<<x.dim_size(i)<<std::endl;
  }
  Status run_sample = sample(&session, seq, seq_pred, inputs, PREDICT_LEN, SEQ_PRE);
  if(!run_sample.ok()){
    LOG(ERROR)<<run_sample;
    return -1;
  }
  std::cout<<"sample finished"<<std::endl;

  saveSequence(output_seq, seq_pred);
  std::cout<<"result saved"<<std::endl;

  session->Close();
}
