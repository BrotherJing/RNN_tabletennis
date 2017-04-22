#ifndef HEADER_MAIN
#define HEADER_MAIN

#define TIMING

tensorflow::Status initialize(std::unique_ptr<tensorflow::Session> *session, std::vector<tensorflow::Tensor> *initialized_outputs);
int sampleTheta(tensorflow::Tensor &theta, int idx);
tensorflow::Status sample(std::unique_ptr<tensorflow::Session> *session, 
  std::vector<CvPoint3D32f> &seq,
  std::vector<CvPoint3D32f> &seq_pred,
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor> > &inputs,
  int predict_len, 
  int seq_pre);
tensorflow::Status LoadGraph(tensorflow::string graph_file_name, std::unique_ptr<tensorflow::Session> *session);
void loadSequence(tensorflow::string input_seq, std::vector<CvPoint3D32f> &seq);
void saveSequence(tensorflow::string output_seq, std::vector<CvPoint3D32f> &seq);
void fillPlaceholder(std::vector<CvPoint3D32f> &seq, tensorflow::Tensor &x, int idx);

#endif