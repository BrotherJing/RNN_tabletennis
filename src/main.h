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

#endif