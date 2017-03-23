#include <fstream>
#include <vector>

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

Status sample(std::unique_ptr<Session> *session, std::vector<std::pair<tensorflow::string, Tensor> > &inputs){
  std::vector<Tensor> outputs;
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
  std::cout<<outputs[4].tensor<float, 3>()(0,0,0)<<std::endl;
  std::cout<<outputs[5].tensor<float, 3>()(0,0,0)<<std::endl;
  std::cout<<outputs[6].tensor<float, 3>()(0,0,0)<<std::endl;
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

int main(int argc, char **argv) {
  string graph = "/home/jing/Documents/RNN_tabletennis/data/export-graph.pb";

  std::vector<Flag> flag_list = {
    Flag("graph", &graph, "graph to be excuted"),
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
  // Initialize a tensorflow session
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

  Tensor x(DT_FLOAT, TensorShape({BATCH_SIZE, NUM_COORDS, SEQUENCE_LENGTH}));
  
  auto tensor = x.tensor<float, 3>();
  tensor(0,0,0) = 0.6;
  tensor(0,1,0) = 0.6;
  tensor(0,2,0) = 0.6;
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
  sample(&session, inputs);
  /*tf::Tensor x(tf::DT_FLOAT, tf::TensorShape()), y(tf::DT_FLOAT, tf::TensorShape());
  x.scalar<float>()() = 23.0;
  y.scalar<float>()() = 19.0;

  std::vector<std::pair<tf::string, tf::Tensor>> input_tensors = {{"x", x}, {"y", y}};
  std::vector<tf::Tensor> output_tensors;

  status = session->Run(input_tensors, {"z"}, {}, &output_tensors);
  checkStatus(status);

  tf::Tensor output = output_tensors[0];
  std::cout << "Success: " << output.scalar<float>() << "!" << std::endl;*/
  session->Close();
}
