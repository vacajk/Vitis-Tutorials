#pragma once
#include <tuple>

#include "constants.h"
#include "types.h"
#include "opencv.hpp"

struct arguments {
  // The path to the input video
  char *input_file;

  // The path to the output video
  char *output_file;

  // The width of the processed image
  int width;

  // The height of the processed image
  int height;

  // The width of the input video
  int in_width;

  // The height of the input video
  int in_height;

  // The number of frames to process
  int nframes;

  // The name of the kernel
  char* kernel_name;

  // The number of compute units on the binary
  int ncompute_units;

  // The path to the xclbin or awsxcbin file
  char* binary_file;

  // If true the output will be converted to grayscale
  bool gray;

  // If true the output will be converted to grayscale
  bool gray_acc;

  // If true prints extra information about the program
  bool verbose;

  bool dec;
  bool enc;
  bool cpu;
};

typedef enum {
    VCU_TYPE_ONLY_FILE,
    VCU_TYPE_ONLY_VCU,
    VCU_TYPE_DEC_FILE,
    VCU_TYPE_FILE_ENC,
    VCU_TYPE_NUM,
} vcu_type_e;


vcu_type_e vcu_type_get(arguments args);

// Parses the command line arguments
arguments parse_args(int argc, char* argv[]);

// Prints the progress of an operation
void print_progress(int cnt, int total);

// Returns an input and output pipe stream.
std::tuple<FILE*, FILE*> get_streams(arguments &args);
std::tuple<cv::VideoCapture*, FILE*> get_streams_cvff(arguments &args);
std::tuple<FILE*, cv::VideoWriter*> get_streams_ffcv(arguments &args);
std::tuple<cv::VideoCapture*, cv::VideoWriter*> get_streams_cv(arguments &args);

// Method for performing convolution
void convolve(cv::VideoCapture* videoIn, cv::VideoWriter* videoOut,
              FILE* streamIn, FILE* streamOut,
              float* coefficients, int coefficient_size,
              arguments args);

void convolve_sw(FILE* streamIn, FILE* streamOut,
              float* coefficients, int coefficient_size,
              arguments args);
