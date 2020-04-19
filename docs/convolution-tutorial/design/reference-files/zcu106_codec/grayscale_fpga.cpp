#include "constants.h"
#include "kernels.h"
#include "types.h"

#include "ap_fixed.h"
#include <hls_stream.h>

#include <cmath>
#include <cstring>

typedef ap_fixed<16,9> fixed;

extern "C"
{

  static const RGBPixel zero = {0, 0, 0, 0};

  void grayscale_read_dataflow(hls::stream<RGBPixel>& read_stream, const RGBPixel *in,
                                 int elements) {
  int pixel = 0;
  while(elements--) {
        read_stream <<  in[pixel++];
  }
}

  void grayscale_compute_dataflow(hls::stream<GrayPixel>& write_stream, hls::stream<RGBPixel>& read_stream,
                                  int elements) {

      RGBPixel pix_rgb;
      GrayPixel pix_gray;

      fixed cr(0.30);
      fixed cg(0.59);
      fixed cb(0.11);

      while(elements--) {
          read_stream >> pix_rgb;

          pix_gray = (pix_rgb.r * cr) + //red
                     (pix_rgb.g * cg) + // green
                     (pix_rgb.b * cb);  // blue

          write_stream << pix_gray;
      }
  }

  void grayscale_write_dataflow(GrayPixel* outFrame, hls::stream<GrayPixel>& write_stream,
                                    int elements) {
    int pixel = 0;
    while(elements--) {
        write_stream >> outFrame[pixel++];
    }
  }

  void grayscale_fpga(const RGBPixel* inFrame, GrayPixel* outFrame,
                     int img_width, int img_height)
  {
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=inFrame bundle=control
#pragma HLS INTERFACE s_axilite port=outFrame bundle=control
#pragma HLS INTERFACE s_axilite port=img_height bundle=control
#pragma HLS INTERFACE s_axilite port=img_width bundle=control
#pragma HLS INTERFACE m_axi port=inFrame bundle=gmem1
#pragma HLS INTERFACE m_axi port=outFrame bundle=gmem2
#pragma HLS data_pack variable=inFrame
#pragma HLS data_pack variable=outFrame

    hls::stream<RGBPixel> read_stream("read");
    hls::stream<GrayPixel> write_stream("write");
    int elements = img_width * img_height;

#pragma HLS dataflow
    grayscale_read_dataflow(read_stream, inFrame, elements);
    grayscale_compute_dataflow(write_stream, read_stream, elements);
    grayscale_write_dataflow(outFrame, write_stream, elements);

  }
}
