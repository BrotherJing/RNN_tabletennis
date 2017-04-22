#ifndef HEADER_TRAJ_PREDICT
#define HEADER_TRAJ_PREDICT

#include <opencv2/opencv.hpp>

#include <Eigen/Eigenvalues> 

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

extern const char* placeholder_x;
extern const char* initial_state_c_0;
extern const char* initial_state_h_0;
extern const char* initial_state_c_1;
extern const char* initial_state_h_1;

extern const char* final_state_c_0;
extern const char* final_state_h_0;
extern const char* final_state_c_1;
extern const char* final_state_h_1;

extern const char* mu1;
extern const char* mu2;
extern const char* mu3;
extern const char* s1;
extern const char* s2;
extern const char* s3;
extern const char* rho;
extern const char* theta;

extern const int BATCH_SIZE;
extern const int SEQUENCE_LENGTH;
extern const int NUM_COORDS;
extern const int SEQ_PRE;
extern const int PREDICT_LEN;

extern const float X_NORM;
extern const float Y_NORM;
extern const float Z_NORM;

struct Coords
{
	float x;
	float y;
	float z;
};

struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](float x) { return dist(gen); });
    }
};

class TrajPredict
{
public:
	TrajPredict(const tensorflow::string graph_file_name);
	~TrajPredict();
	tensorflow::Status sample(std::vector<CvPoint3D32f> &seq,
	  std::vector<CvPoint3D32f> &seq_pred,
	  int predict_len, 
	  int sl_pre);
    tensorflow::Status sampleN(CvPoint3D32f coord,
      std::vector<CvPoint3D32f> &seq_pred,
      int predict_len);
    CvPoint3D32f sample1(CvPoint3D32f coord);
    bool clearState();
	
private:
	tensorflow::Status initialize(std::unique_ptr<tensorflow::Session> *session, std::vector<tensorflow::Tensor> *initialized_outputs);
	int sampleTheta(tensorflow::Tensor &theta, int idx);
    tensorflow::Status LoadGraph(const tensorflow::string graph_file_name, std::unique_ptr<tensorflow::Session> *session);
	void fillPlaceholder(CvPoint3D32f coord, tensorflow::Tensor &x);

	std::unique_ptr<tensorflow::Session> session;
	std::vector<tensorflow::Tensor> initialized_outputs;
  	std::vector<std::pair<tensorflow::string, tensorflow::Tensor> > inputs;
};

#endif