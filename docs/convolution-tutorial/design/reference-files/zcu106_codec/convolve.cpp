
#include "common.h"
#include "constants.h"
#include "kernels.h"

#include <vector>
#include <cstdio>

#include "xcl2.hpp"

using std::vector;

bool operator==(const RGBPixel& lhs, const RGBPixel& rhs) {
    return lhs.r == rhs.r &&
           lhs.g == rhs.g &&
           lhs.b == rhs.b;
}

void
test(vector<RGBPixel>& in, vector<RGBPixel>& out,
     float* coefficients, int coefficient_size,
     int width, int height) {
    vector<RGBPixel> gold(out.size());
    convolve_cpu(in.data(), gold.data(), coefficients, coefficient_size, width, height);
    auto it = mismatch(begin(gold), end(gold), begin(out));
    if(it.first != end(gold)) {
        printf("Incorrect result: \n Expected: (%d %d %d)\nResult:  (%d %d %d)\n ",
               it.first->r, it.first->g, it.first->b, it.second->r, it.second->g, it.second->b);
    }
}

void convolve(cv::VideoCapture* videoIn, cv::VideoWriter* videoOut,
              FILE* streamIn, FILE* streamOut,
              float* coefficients, int coefficient_size,
              arguments args) {
    size_t frame_bytes = args.width * args.height * sizeof(RGBPixel);
    size_t gray_frame_bytes = args.width * args.height * sizeof(GrayPixel);
    vector<RGBPixel> inFrame(args.width * args.height);
    vector<RGBPixel> outFrame(args.width * args.height);
    vector<GrayPixel> grayFrame(args.width * args.height);

    bool frame_read = false;
    size_t bytes_read = 0;
    size_t bytes_written = 0;

    size_t total_coefficient_size = coefficient_size * coefficient_size;
    vector<float, aligned_allocator<float>> filter_coeff(coefficients, coefficients + total_coefficient_size);
    size_t coefficient_size_bytes = sizeof(float) * total_coefficient_size;


    vector<cl::Device> devices = xcl::get_xil_devices();
    std::cout << "devices number : " << devices.size() << std::endl;
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device,
    cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);

    cl::Program::Binaries bins = xcl::import_binary_file(args.binary_file);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    cl::Kernel convolve_kernel(program, "convolve_fpga");
    cl::Kernel grayscale_kernel(program, "grayscale_fpga");

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, frame_bytes, NULL);
    cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, frame_bytes, NULL);
    cl::Buffer buffer_gary(context, CL_MEM_WRITE_ONLY, gray_frame_bytes, NULL);
    cl::Buffer buffer_coefficient(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, coefficient_size_bytes, filter_coeff.data());

    int compute_units = 1;
    int lines_per_compute_unit = args.height / compute_units;

    convolve_kernel.setArg(0, buffer_input);
    convolve_kernel.setArg(1, buffer_output);
    convolve_kernel.setArg(2, buffer_coefficient);
    convolve_kernel.setArg(3, coefficient_size);
    convolve_kernel.setArg(4, args.width);
    convolve_kernel.setArg(5, args.height);
    convolve_kernel.setArg(6, 0);
    convolve_kernel.setArg(7, lines_per_compute_unit);

    if(args.gray && args.gray_acc)
    {
        grayscale_kernel.setArg(0, buffer_output);
        grayscale_kernel.setArg(1, buffer_gary);
        grayscale_kernel.setArg(2, args.width);
        grayscale_kernel.setArg(3, args.height);
    }

    q.enqueueMigrateMemObjects({buffer_coefficient}, 0);

    std::cout << "            compute_units = " << compute_units << " " << args.ncompute_units << std::endl;
    std::cout << "   lines_per_compute_unit = " << lines_per_compute_unit << std::endl;
    std::cout << "                     gray = " << args.gray << " " << args.gray_acc << std::endl;
    
    auto fpga_begin = std::chrono::high_resolution_clock::now();

    cv::Mat inFrameMatNV12;
    cv::Mat inFrameMatRGBA(args.height, args.width, CV_8UC4);
    cv::Mat outFrameMatGRAY(args.height, args.width, CV_8UC1);
    cv::Mat outFrameMatRGBA(args.height, args.width, CV_8UC4);
    cv::Mat outFrameMatNV12(args.height, args.width, CV_8UC2);
    cv::Mat outFrameMatRGB(args.height, args.width, CV_8UC3);

    void *inFrameDataRGBA = (void *)inFrameMatRGBA.data;
    void *outFrameData = nullptr;

    for(int frame_count = 0; frame_count < args.nframes; frame_count++) {
        // Read frame
        if(args.dec)
        {
            frame_read = videoIn->read(inFrameMatNV12);
            if(!frame_read) {
                printf("\nError: partial frame %d read failed", frame_count);
                break;
            }
            if(frame_count == 0)
            {
                printf("inFrameMatNV12 %d %d %d %d\n", inFrameMatNV12.cols, inFrameMatNV12.rows, inFrameMatNV12.channels(), inFrameMatNV12.type());
            }
            cv::cvtColor(inFrameMatNV12, inFrameMatRGBA, CV_YUV2RGBA_NV12);
        }
        else
        {
            bytes_read = fread(inFrameDataRGBA, 1, frame_bytes, streamIn);
            if(bytes_read != frame_bytes) {
                printf("\nError: partial frame.\nExpected %zu\nActual %zu\n", frame_bytes, bytes_read);
                break;
            }
        }

        q.enqueueWriteBuffer(buffer_input, CL_FALSE, 0, frame_bytes, inFrameDataRGBA);
        q.enqueueTask(convolve_kernel);
        if(args.gray) {
            outFrameData = (void *)outFrameMatGRAY.data;

            if(args.gray_acc)
            {
                q.enqueueTask(grayscale_kernel);
                q.enqueueReadBuffer(buffer_gary, CL_TRUE, 0, gray_frame_bytes, outFrameData);
            }
            else
            {
                q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, frame_bytes, outFrame.data());
                grayscale_cpu(outFrame.data(), (GrayPixel *)outFrameData, args.width, args.height);
            }

            // Write frame
            if(args.enc)
            {
                //encoder does not support gray, we can convert GRAY2RGB within OpenCV or Gstreamer
                //cv::cvtColor(outFrameMatGRAY, outFrameMatRGB, CV_GRAY2RGB);
                videoOut->write(outFrameMatGRAY);
                if(frame_count == 0)
                {
                    printf("outFrameMatGRAY %d %d %d %d\n", outFrameMatGRAY.cols, outFrameMatGRAY.rows, outFrameMatGRAY.channels(), outFrameMatGRAY.type());
                }
            }
            else
            {
                bytes_written = fwrite(outFrameData, 1, gray_frame_bytes, streamOut);
                fflush(streamOut);
                if (bytes_written != gray_frame_bytes) {
                    printf("\nError: partial frame.\nExpected %zu\nActual %zu\n",
                           gray_frame_bytes, bytes_written);
                    break;
                }
            }
        } else {
            outFrameData = (void *)outFrameMatRGBA.data;

            q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, frame_bytes, outFrameData);

            if(args.enc)
            {
                //encoder does not support rgba, we can convert RGBA2BGR within OpenCV or Gstreamer
                cv::cvtColor(outFrameMatRGBA, outFrameMatRGB, CV_RGBA2BGR);
                videoOut->write(outFrameMatRGB);

                if(frame_count == 0)
                {
                    printf("outFrameMatRGBA %d %d %d %d\n", outFrameMatRGB.cols, outFrameMatRGB.rows, outFrameMatRGB.channels(), outFrameMatRGB.type());
                }
            }
            else
            {
                bytes_written = fwrite(outFrameData, 1, frame_bytes, streamOut);
                fflush(streamOut);
                if (bytes_written != frame_bytes) {
                    printf("\nError: partial frame.\nExpected %zu\nActual %zu\n",
                           frame_bytes, bytes_written);
                    break;
                }
            }
            // test(inFrame, outFrame, coefficients, coefficient_size, width, height);
        }

        print_progress(frame_count, args.nframes);
    }
    q.finish();

    auto fpga_end = std::chrono::high_resolution_clock::now();

    // Report performance (if not running in emulation mode)
    if (getenv("XCL_EMULATION_MODE") == NULL) {
        std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
        std::cout << "                 " << std::endl;
        std::cout << "FPGA Time:       " << fpga_duration.count() << " s" << std::endl;
        std::cout << "FPGA Throughput: "
                  << (1920*1080*4*132) / fpga_duration.count() / (1024.0*1024.0)
                  << " MB/s" << std::endl;
     }
}

void convolve_sw(FILE* streamIn, FILE* streamOut,
              float* coefficients, int coefficient_size,
              arguments args) {
    size_t frame_bytes = args.width * args.height * sizeof(RGBPixel);
    size_t gray_frame_bytes = args.width * args.height * sizeof(GrayPixel);
    vector<RGBPixel> inFrame(args.width * args.height);
    vector<RGBPixel> outFrame(args.width * args.height);
    vector<GrayPixel> grayFrame(args.width * args.height);

    size_t bytes_read = 0;
    size_t bytes_written = 0;
    for(int frame_count = 0; frame_count < args.nframes; frame_count++) {
        // Read frame
        bytes_read = fread(inFrame.data(), 1, frame_bytes, streamIn);
        if(bytes_read != frame_bytes) {
            printf("\nError: partial frame.\nExpected %zu\nActual %zu\n", frame_bytes, bytes_read);
            break;
        }

        convolve_cpu(inFrame.data(), outFrame.data(),
                     coefficients, coefficient_size,
                     args.width, args.height);

        if(args.gray) {
          grayscale_cpu(outFrame.data(), grayFrame.data(), args.width, args.height);
          bytes_written = fwrite(grayFrame.data(), 1, gray_frame_bytes, streamOut);
          fflush(streamOut);
          if (bytes_written != gray_frame_bytes) {
            printf("\nError: partial frame.\nExpected %zu\nActual %zu\n",
                   gray_frame_bytes, bytes_written);
            break;
          }
        } else {
          bytes_written = fwrite(outFrame.data(), 1, frame_bytes, streamOut);
          fflush(streamOut);
          if (bytes_written != frame_bytes) {
            printf("\nError: partial frame.\nExpected %zu\nActual %zu\n",
                   frame_bytes, bytes_written);
            break;
          }
          // test(inFrame, outFrame, coefficients, coefficient_size, width, height);
        }

        print_progress(frame_count, args.nframes);
    }
}
