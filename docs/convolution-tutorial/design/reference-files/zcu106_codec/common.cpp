
#include <cstdio>
#include <cstring>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <argp.h>
#include "common.h"
#include "kernels.h"

using std::make_tuple;
using std::tuple;
using std::string;
using std::vector;
using std::tie;

// Program documentation.
static char doc[] = "Xilinx FPGA Convolution Tutorial";

// A description of the arguments we accept.
static char args_doc[] = "XCL_BINARY_FILE INPUT_FILE";

static char default_kernel_name[] = "";

// The options we understand.
static struct argp_option options[] = {
    {"gray", 'g', 0, 0, "Convert input to grayscale"},
    {"gray_acc", 'a', 0, 0, "Accelerate grayscale"},
    {"output", 'o', "FILE", 0, "Output file (default: output.mp4)"},
    {"scale", 's', "WIDTH HEIGHT", 0,
     "The input will be resized to this before being processed"},
    {"quiet", 'q', 0, 0, "Display verbose output"},
    {"nframes", 'n', "NUM", 0, "Number of frames to process"},
    {"kernel_name", 'k', "KERNEL_NAME", 0, "The kernel to launch"},
    {"ncomputeunits", 'c', "NUM", 0, "Number of compute units"},
    {"dec", 'd', 0, 0, "Enable VCU decoder"},
    {"enc", 'e', 0, 0, "Enable VCU encoder"},
    {"cpu", 'u', 0, 0, "Enable CPU only"},
    {0}};

char default_output[] = "output.mp4";
const char bar[] = "###################################";

namespace {

tuple<int, int, int>
get_video_size(string path) {
  string command("ffprobe -v error -show_entries stream=width,height,nb_frames -of default=nw=1:nk=1:nokey=1:noprint_wrappers=1 " + path);
  FILE* size_stream = popen(command.c_str(), "r");
  int width = 0, height = 0, frames=0;
  fscanf(size_stream, "%d", &width);
  fscanf(size_stream, "%d", &height);
  fscanf(size_stream, "%d", &frames);
  pclose(size_stream);

  return make_tuple(width, height, frames);
}

// Parse a single option.
static error_t
parse_opt (int key, char *arg, struct argp_state *state) {

  struct arguments *arguments = (struct arguments*)state->input;

  switch (key) {
    case 'g':
      arguments->gray = true;
      break;

    case 'a':
      arguments->gray_acc = true;
      break;

    case 'd':
      arguments->dec = true;
      break;

    case 'e':
      arguments->enc = true;
      break;

    case 'u':
      arguments->cpu = true;
      break;

    case 'o':
      arguments->output_file = arg;
      break;

    case 's':
      arguments->width = atoi(arg);
      arguments->height = atoi(state->argv[state->next]);
      state->next++;
      break;

    case 'n':
      arguments->nframes = atoi(arg);
      break;

    case 'q':
      arguments->verbose = false;
      break;

    case 'k':
      arguments->kernel_name = arg;
      break;

    case 'c':
      arguments->ncompute_units = atoi(arg);
      break;

    case ARGP_KEY_ARG:
      if(strstr(arg, "xclbin")) {
        arguments->binary_file = arg;
      } else {
        arguments->input_file = arg;
      }
      break;

    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

}  // anonymous namespace

static struct argp argp = { options, parse_opt, args_doc, doc };

void print_progress(int cnt, int total) {
    int bar_length = 35;

    int elements = cnt+1;
    float percent = float(elements)/(total);
    int bar_elements = float(bar_length) * percent;

    printf("\r[%-*.*s] %3.f %%", bar_length, bar_elements, &(bar[bar_length - bar_elements]), percent * 100);
    fflush(stdout);
}

vcu_type_e vcu_type_get(arguments args)
{
    if(!args.dec && !args.enc)
        return VCU_TYPE_ONLY_FILE;
    else if(args.dec && !args.enc)
        return VCU_TYPE_DEC_FILE;
    else if(!args.dec && args.enc)
        return VCU_TYPE_FILE_ENC;
    else if(args.dec && args.enc)
        return VCU_TYPE_ONLY_VCU;
    else
        return VCU_TYPE_NUM;
}

arguments parse_args(int argc, char* argv[]) {
    arguments arguments;

    arguments.output_file = default_output;
    arguments.gray = false;
    arguments.gray_acc = false;
    arguments.verbose = true;
    arguments.width  = -1;
    arguments.height = -1;
    arguments.nframes = -1;
    arguments.binary_file = nullptr;
    arguments.kernel_name = default_kernel_name;
    arguments.ncompute_units = 1;
    arguments.dec = false;
    arguments.enc = false;
    arguments.cpu = false;

    argp_parse (&argp, argc, argv, 0, 0, &arguments);

    int input_width, input_height, input_frames;
    tie(input_width, input_height, input_frames) = get_video_size(arguments.input_file);
    arguments.in_width = input_width;
    arguments.in_height = input_height;

    if(arguments.width == -1)   { arguments.width  = input_width; }
    if(arguments.height == -1)  { arguments.height  = input_height; }
    if(arguments.nframes == -1) { arguments.nframes = input_frames; }

    if (arguments.verbose) {
      printf("input: %s\noutput: %s\n", arguments.input_file, arguments.output_file);
      printf("video size: %dx%d\n", input_width, input_height);
      if(arguments.width == -1 || arguments.height == -1) {
        printf("scaled size: %dx%d\n", arguments.width, arguments.height);
      }
      printf("nframes: %d\n", arguments.nframes);
    }

    return arguments;
}

tuple<FILE*, FILE*>
get_streams(arguments &args) {
  string inCommand;
  inCommand.resize(2048);
  snprintf(&inCommand.front(), 2048,
           "ffmpeg -v error -hide_banner -i %s -f image2pipe -vcodec rawvideo -vf scale=w=%d:h=%d -vframes %d -pix_fmt rgba -",
           args.input_file, args.width, args.height, args.nframes);

  string outCommand;
  outCommand.resize(2048);
  snprintf(&outCommand.front(), 2048,
           "ffmpeg -v error -hide_banner -y -f rawvideo -vcodec rawvideo -pix_fmt %s -s %dx%d -framerate 25 -i - -f mp4 -q:v 5 -an -codec mpeg4 %s",
           (args.gray) ? "gray" : "rgba", args.width, args.height, args.output_file);

  if(args.verbose) {
    printf("IN COMMAND:  %s\n"
           "OUT COMMAND: %s\n", inCommand.c_str(), outCommand.c_str());
    if(args.binary_file) printf("Binary Path: %s\n", args.binary_file);
  }

  FILE* streamIn = popen(inCommand.c_str(), "r");
  FILE* streamOut = popen(outCommand.c_str(), "w");

  return make_tuple(streamIn, streamOut);
}

//#include "opencv2/videoio.hpp"

tuple<cv::VideoCapture*, FILE*>
get_streams_cvff(arguments &args) {
    string inCommand;
    inCommand.resize(2048);
    snprintf(&inCommand.front(), 2048,
             "gst-launch-1.0 filesrc location=%s ! qtdemux ! queue ! h264parse ! video/x-h264, alignment=au ! omxh264dec ! video/x-raw,format=NV12,width=%d,height=%d ! appsink",
             args.input_file, args.width, args.height);

    string outCommand;
    outCommand.resize(2048);
    snprintf(&outCommand.front(), 2048,
             "ffmpeg -v error -hide_banner -y -f rawvideo -vcodec rawvideo -pix_fmt %s -s %dx%d -framerate 25 -i - -f mp4 -q:v 5 -an -codec mpeg4 %s",
             (args.gray) ? "gray" : "rgba", args.width, args.height, args.output_file);

    if(args.verbose) {
        printf("IN COMMAND:  %s\n"
               "OUT COMMAND: %s\n", inCommand.c_str(), outCommand.c_str());
        if(args.binary_file) printf("Binary Path: %s\n", args.binary_file);
    }

    cv::VideoCapture *cap = new cv::VideoCapture(inCommand, cv::CAP_GSTREAMER);
    FILE* streamOut = popen(outCommand.c_str(), "w");

    if(!cap->isOpened())
    {
        printf("\nError: create VideoCapture failed!\n");
    }

    return make_tuple(cap, streamOut);
}

tuple<FILE*, cv::VideoWriter*>
get_streams_ffcv(arguments &args) {
    string inCommand;
    inCommand.resize(2048);
    snprintf(&inCommand.front(), 2048,
             "ffmpeg -v error -hide_banner -i %s -f image2pipe -vcodec rawvideo -vf scale=w=%d:h=%d -vframes %d -pix_fmt rgba -",
             args.input_file, args.width, args.height, args.nframes);

    string outCommand;
    outCommand.resize(2048);
#if 0
    //force to NV12
    snprintf(&outCommand.front(), 2048,
            "appsrc ! videoconvert ! video/x-raw,format=NV12,width=%d,height=%d "
            "! omxh264enc target-bitrate=2000 ! video/x-h264, alignment=au"
            "! queue ! capsfilter ! h264parse ! queue ! qtmux ! filesink location=%s ",
            args.width, args.height, args.output_file);
#else
    snprintf(&outCommand.front(), 2048,
            "appsrc "
            "! queue ! videoconvert ! video/x-raw,width=%d,height=%d "
            "! queue ! omxh264enc target-bitrate=2000 ! video/x-h264, alignment=au "
            "! queue ! capsfilter ! h264parse "
            "! queue ! qtmux "
            "! queue ! filesink location=%s",
            args.width, args.height, args.output_file);
#endif

    if(args.verbose) {
        printf("IN COMMAND:  %s\n"
               "OUT COMMAND: %s\n", inCommand.c_str(), outCommand.c_str());
        if(args.binary_file) printf("Binary Path: %s\n", args.binary_file);
    }

//    int fourcc = args.gray? CV_FOURCC('Y','8','0','0'): CV_FOURCC('N','V','1','2');
//    printf("CV_FOURCC: 0x%08x\n", fourcc);
    FILE* streamIn = popen(inCommand.c_str(), "r");
    cv::VideoWriter *vwr = new cv::VideoWriter(outCommand, cv::CAP_GSTREAMER, 0, (double)25, cv::Size(args.width, args.height), (args.gray)? false: true);

    if(!vwr->isOpened())
    {
        printf("\nError: create VideoWriter failed!\n");
    }

    return make_tuple(streamIn, vwr);
}

tuple<cv::VideoCapture*, cv::VideoWriter*>
get_streams_cv(arguments &args) {
    string inCommand;
    inCommand.resize(2048);
    snprintf(&inCommand.front(), 2048,
             "filesrc location=%s "
             "! queue ! qtdemux "
             "! queue ! h264parse ! video/x-h264, alignment=au "
             "! queue ! omxh264dec ! video/x-raw,format=NV12,width=%d,height=%d "
             "! queue ! appsink",
             args.input_file, args.width, args.height);

    string outCommand;
    outCommand.resize(2048);
#if 0
    //force to NV12
    snprintf(&outCommand.front(), 2048,
            "appsrc ! videoconvert ! video/x-raw,format=NV12,width=%d,height=%d "
            "! omxh264enc target-bitrate=2000 ! video/x-h264, alignment=au"
            "! queue ! capsfilter ! h264parse ! queue ! qtmux ! filesink location=%s ",
            args.width, args.height, args.output_file);
#else
    snprintf(&outCommand.front(), 2048,
            "appsrc "
            "! queue ! videoconvert ! video/x-raw,width=%d,height=%d "
            "! queue ! omxh264enc target-bitrate=2000 ! video/x-h264, alignment=au "
            "! queue ! capsfilter ! h264parse "
            "! queue ! qtmux "
            "! queue ! filesink location=%s",
            args.width, args.height, args.output_file);
#endif

    if(args.verbose) {
        printf("IN COMMAND:  %s\n"
               "OUT COMMAND: %s\n", inCommand.c_str(), outCommand.c_str());
        if(args.binary_file) printf("Binary Path: %s\n", args.binary_file);
    }

    //int fourcc = args.gray? CV_FOURCC('Y','8','0','0'): CV_FOURCC('N','V','1','2');
    //printf("CV_FOURCC: 0x%08x\n", fourcc);
    cv::VideoCapture *cap = new cv::VideoCapture(inCommand, cv::CAP_GSTREAMER);
    cv::VideoWriter *vwr = new cv::VideoWriter(outCommand, cv::CAP_GSTREAMER, 0, (double)25, cv::Size(args.width, args.height), (args.gray)? false: true);

    if(!cap->isOpened())
    {
        printf("\nError: create VideoCapture failed!\n");
    }

    if(!vwr->isOpened())
    {
        printf("\nError: create VideoWriter failed!\n");
    }

    return make_tuple(cap, vwr);
}
