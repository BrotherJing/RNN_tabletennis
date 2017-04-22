#include <fstream>
#include <iostream>
#include <cmath>

#include <stdlib.h>//srand, rand
#include <time.h>//time
#include <sys/time.h>//for timing

#include <opencv2/opencv.hpp>

#include "TrajPredict.h"
#include "main.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flags;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

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
  std::vector<CvPoint3D32f> &seq,
  std::vector<CvPoint3D32f> &seq_pred,
  std::vector<std::pair<tensorflow::string, Tensor> > &inputs,
  int predict_len, 
  int sl_pre){

  double bias = 1.0;
  double mean[3];
  double stderr[4];

  std::vector<CvPoint3D32f> seq_feed(predict_len+1);
  for(int i=0;i<sl_pre+1;++i){
    seq_feed[i] = seq[i];
  }

  std::vector<Tensor> outputs;

#ifdef TIMING
  struct timeval t1,t2;
  double timeuse;
#endif

  for(int i=0;i<predict_len;++i){

#ifdef TIMING
    gettimeofday(&t1,NULL);
#endif
    
    fillPlaceholder(seq_feed, inputs[4].second, i);
  
    Status run_sample = (*session)->Run(inputs, {
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
      seq_feed[i+1] = CvPoint3D32f{seq_feed[i].x+float(draw(0)), seq_feed[i].y+float(draw(1)), seq_feed[i].z+float(draw(2))};

#ifdef TIMING
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
    printf("Use Time:%fms\n",timeuse);
#endif
  }
  for(int i=0;i<predict_len;++i){
    seq_pred.push_back(CvPoint3D32f{seq_feed[i].x*X_NORM, seq_feed[i].y*Y_NORM, seq_feed[i].z*Z_NORM});
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

void loadSequence(string input_seq, std::vector<CvPoint3D32f> &seq){
  float x,y,z,max_z=0;
  std::ifstream input;
  input.open(input_seq.c_str(), std::ifstream::in);
  while(!input.eof()){
    input>>x>>y>>z;
    CvPoint3D32f coord{x/X_NORM,y/Y_NORM,z/Z_NORM};
    seq.push_back(coord);
  }
  input.close();
}

void saveSequence(string output_seq, std::vector<CvPoint3D32f> &seq){
  std::ofstream output;
  output.open(output_seq.c_str(), std::ios::out|std::ios::trunc);
  for(int i=0;i<seq.size();++i){
    output<<seq[i].x<<" "<<seq[i].y<<" "<<seq[i].z<<std::endl;
  }
  output.close();
}

void fillPlaceholder(std::vector<CvPoint3D32f> &seq, Tensor &x, int idx){
  auto tensor = x.tensor<float, 3>();
  CvPoint3D32f coord = seq[idx];
  tensor(0,0,0) = coord.x;
  tensor(0,1,0) = coord.y;
  tensor(0,2,0) = coord.z;
}

int main(int argc, char **argv) {

  srand(time(NULL));

  string graph = "/home/jing/Documents/RNN_tabletennis/data/export-graph.pb";
  string input_seq = "input.csv";
  string output_seq = "output.csv";

  TrajPredict pred(graph);
  std::vector<CvPoint3D32f> seq;
  std::vector<CvPoint3D32f> seq_pred;
  loadSequence(input_seq, seq);
  Status run_sample = pred.sample(seq, seq_pred, PREDICT_LEN, SEQ_PRE);
  if(!run_sample.ok()){
    LOG(ERROR)<<run_sample;
    return -1;
  }
  std::cout<<"sample finished"<<std::endl;
  saveSequence(output_seq, seq_pred);
  std::cout<<"result saved"<<std::endl;
  /*std::vector<Flag> flag_list = {
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

  Tensor x(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_COORDS, 1}));

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

  session->Close();*/
}
