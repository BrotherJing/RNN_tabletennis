#ifndef HEADER_MAIN
#define HEADER_MAIN

const char* placeholder_x = "Test/Model/Input_data:0";
const char* initial_state_c_0 = "Test/Model/zeros:0";
const char* initial_state_h_0 = "Test/Model/zeros_1:0";
const char* initial_state_c_1 = "Test/Model/zeros_2:0";
const char* initial_state_h_1 = "Test/Model/zeros_3:0";

const char* final_state_c_0 = "Test/Model/RNN/multi_rnn_cell_59/cell_0/lstm_cell/add_3:0";
const char* final_state_h_0 = "Test/Model/RNN/multi_rnn_cell_59/cell_0/lstm_cell/mul_5:0";
const char* final_state_c_1 = "Test/Model/RNN/multi_rnn_cell_59/cell_1/lstm_cell/add_3:0";
const char* final_state_h_1 = "Test/Model/RNN/multi_rnn_cell_59/cell_1/lstm_cell/mul_5:0";

const char* mu1 = "Test/Model/MDN/split_1:0";
const char* mu2 = "Test/Model/MDN/split_1:1";
const char* mu3 = "Test/Model/MDN/split_1:2";
const char* s1 = "Test/Model/MDN/Exp_1:0";
const char* s2 = "Test/Model/MDN/Exp_2:0";
const char* s3 = "Test/Model/MDN/Exp_3:0";
const char* rho = "Test/Model/MDN/Tanh:0";
const char* theta = "Test/Model/MDN/Mul:0";

const int BATCH_SIZE = 64;
const int SEQUENCE_LENGTH = 60;
const int NUM_COORDS = 3;
const int SEQ_PRE = 30;
const int PREDICT_LEN = 100;

const float X_NORM = 1525;
const float Y_NORM = 2740;
const float Z_NORM = 458;

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

Status initialize(std::unique_ptr<Session> *session, std::vector<Tensor> *initialized_outputs);
int sampleTheta(Tensor &theta, int idx);
Status sample(std::unique_ptr<Session> *session, 
  std::vector<Coords> &seq,
  std::vector<Coords> &seq_pred,
  std::vector<std::pair<tensorflow::string, Tensor> > &inputs,
  int predict_len, 
  int seq_pre);
Status LoadGraph(string graph_file_name, std::unique_ptr<Session> *session);
void loadSequence(string input_seq, std::vector<Coords> &seq);
void saveSequence(string output_seq, std::vector<Coords> &seq);
void fillPlaceholder(std::vector<Coords> &seq, Tensor &x, int offset, int length);

#endif