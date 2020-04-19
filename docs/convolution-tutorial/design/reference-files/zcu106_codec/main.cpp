
#include "common.h"
#include "constants.h"
#include "filters.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <tuple>

using std::chrono::duration;
using std::chrono::system_clock;
using std::tie;

int main(int argc, char* argv[]) {
    // Parse command line
    arguments opt = parse_args(argc, argv);
    int input_size = opt.width * opt.height * sizeof(RGBPixel);

    cv::VideoCapture *videoIn = nullptr;
    cv::VideoWriter *videoOut = nullptr;
    FILE *streamIn = nullptr;
    FILE *streamOut = nullptr;

    printf("Accel:%s\n", opt.cpu?"OFF":"ON");
    printf("VCU decoder:%s encoder:%s\n", opt.dec?"ON":"OFF", opt.enc?"ON":"OFF");

    vcu_type_e vcu_type = vcu_type_get(opt);
    switch(vcu_type)
    {
    case VCU_TYPE_ONLY_FILE:
        tie(streamIn, streamOut) = get_streams(opt);
        break;
    case VCU_TYPE_ONLY_VCU:
        tie(videoIn, videoOut) = get_streams_cv(opt);
        break;
    case VCU_TYPE_DEC_FILE:
        tie(videoIn, streamOut) = get_streams_cvff(opt);
        break;
    case VCU_TYPE_FILE_ENC:
        tie(streamIn, videoOut) = get_streams_ffcv(opt);
        break;
    default:
        return EXIT_FAILURE;
        break;
    }

    float* coefficients = gaussian;
    int coefficient_size = 3;

    printf("Processing %d frames of %s ...\n", opt.nframes, opt.input_file);

    auto start = system_clock::now();
    if(opt.cpu)
    {
        convolve_sw(streamIn, streamOut, coefficients, coefficient_size, opt);
    }
    else
    {
        convolve(videoIn, videoOut, streamIn, streamOut, coefficients, coefficient_size, opt);
    }
    float elapsed = duration<float>(system_clock::now() - start).count();

    if(videoIn) videoIn->release();
    if(videoOut) videoOut->release();
    if(streamIn) fflush(streamIn);
    if(streamOut) fflush(streamOut);

    double mbps = opt.nframes * input_size / 1024. / 1024. / elapsed;
    printf("\n\nProcessed %2.2f MB in %3.3fs (%3.2f MBps)\n\n",
           input_size / 1024. /1024., elapsed, mbps);

    return EXIT_SUCCESS;
}
