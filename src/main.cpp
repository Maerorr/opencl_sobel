#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <iostream>
#include <CL/cl.hpp>
#include <array>
#include <fstream>
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CL_CHECK(x, name) \
    if (x != CL_SUCCESS) { \
        std::cerr << name << " - Error: " << x << std::endl; \
        return 1; \
    }

int main() {
    // prepare opencl context
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();
    auto context = cl::Context(device);
    auto queue = cl::CommandQueue(context, device);

    // load image using stbi
    int width, height, channels;
    unsigned char *image = stbi_load("res/input/sobel_test.jpg", &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << "x" << channels << std::endl;

    // convert image to greyscale float
    // float *greyscale_image = new float[width * height];
    // for (int i = 0; i < width * height; i++) {
    //     greyscale_image[i] = (image[i * 3] + image[i * 3 + 1] + image[i * 3 + 2]) / 3.0f;
    // }

    // add alpha channel
    float *rgba_image = new float[width * height * 4];
    for (int i = 0; i < width * height; i++) {
        rgba_image[i * 4] = image[i * 3] / 255.0f;
        rgba_image[i * 4 + 1] = image[i * 3 + 1] / 255.0f;
        rgba_image[i * 4 + 2] = image[i * 3 + 2] / 255.0f;
        rgba_image[i * 4 + 3] = 1.0f;
    }

    // unsigned char *test = new unsigned char[width * height * 4];
    // for (int i = 0; i < width * height * 4; i++) {
    //     test[i] = rgba_image[i] * 255;
    // }

    // stbi_write_png("res/output/read.png", width, height, 4, test, width * 4);

    cl_int err;
    // create image2d
    auto input_image = cl::Image2D(
        context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cl::ImageFormat(CL_RGBA, CL_FLOAT), 
        width, 
        height, 
        0, 
        rgba_image,
        &err);
    CL_CHECK(err, "create input 2d image");

    // create output image
    auto output_image = cl::Image2D(context, CL_MEM_WRITE_ONLY,
                                   cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                   width, height, 0, nullptr, &err);

    CL_CHECK(err, "create output 2d image");

    cl::size_t<3> origin = {};
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region = {};
    region[0] = width;
    region[1] = height;
    region[2] = 1;
    
    int filter_width = 3;
    int filter_size = filter_width * filter_width;

    // create filter
    float sobel_x[9] = {
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1
    };

    float sobel_y[9] = {
            -1, -2, -1,
             0,  0,  0,
             1,  2,  1
    };

    // create filter buffer
    auto sobel_x_filter = cl::Buffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * filter_size, NULL, &err);
    CL_CHECK(err, "create sobel x filter buffer");

    auto sobel_y_filter = cl::Buffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * filter_size, NULL, &err);
    CL_CHECK(err, "create sobel y filter buffer");

    // write filter buffers
    CL_CHECK(
        queue.enqueueWriteBuffer(
            sobel_x_filter,
            CL_TRUE,
            0,
            sizeof(float) * filter_size,
            sobel_x),
            "write sobel x filter");
    CL_CHECK(
        queue.enqueueWriteBuffer(
            sobel_y_filter,
            CL_TRUE,
            0,
            sizeof(float) * filter_size,
            sobel_y),
            "write sobel y filter");

    // create image sampler
    auto sampler = cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

    // load kernel source
    std::ifstream kernel_file("res/kernels/sobel_kernel.cl");
    std::string source_code( std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(source_code.c_str(), source_code.length()));

    // create program
    auto program = cl::Program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // create kernel
    auto kernel = cl::Kernel(program, "convolution", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return 1;
    }
    // set kernel arguments
    std::cout << "settings args" << std::endl;
    CL_CHECK(kernel.setArg(0, input_image), "setting arg 0 (input image)");
    CL_CHECK(kernel.setArg(1, output_image), "setting arg 1 (output image)");
    CL_CHECK(kernel.setArg(2, sampler), "setting arg 2 (sampler)");
    CL_CHECK(kernel.setArg(3, sobel_x_filter), "setting arg 3 (sobel x filter)");
    CL_CHECK(kernel.setArg(4, sobel_y_filter), "setting arg 4 (sobel y filter)");
    CL_CHECK(kernel.setArg(5, 0), "setting arg 5 (x or y)");
    CL_CHECK(kernel.setArg(6, filter_width), "setting arg 6 (filter width)");

//    kernel.setArg(3, &sobel_x_filter);
//    kernel.setArg(4, &sobel_y_filter);
//    kernel.setArg(5, 0);
//    kernel.setArg(6, filter_width);
//
//    __constant float *sobel_x,
//    __constant float *sobel_y,
//    int x_or_y,
//    int filter_width,
//    sampler_t sampler

    // execute kernel
    CL_CHECK(queue.enqueueNDRangeKernel(
        kernel, 
        cl::NullRange, 
        cl::NDRange(width, height),
        cl::NullRange), 
        "starting kernel");

    output_image.getImageInfo(CL_IMAGE_WIDTH, &width);
    output_image.getImageInfo(CL_IMAGE_HEIGHT, &height);
    output_image.getImageInfo(CL_IMAGE_DEPTH, &channels);
    std::cout << "Image width: " << width << std::endl;
    std::cout << "Image height: " << height << std::endl;
    std::cout << "Image channels: " << channels << std::endl;

    float *sobel_x_read = new float[filter_size];
    CL_CHECK(
            queue.enqueueReadBuffer(
                    sobel_x_filter,
                    CL_TRUE,
                    0,
                    sizeof(float) * filter_size,
                    sobel_x_read
                    ),
                    "reading sobel x filter"
            );

    // read image
    float *output = new float[width * height * 4];
    CL_CHECK(
        queue.enqueueReadImage(
            output_image, 
            CL_TRUE, 
            origin, 
            region, 
            0, 
            0, 
            output), 
            "read output image");

    // display the output image
    for (int i = 0; i < 16; i += 4) {
        std::cout << "(" << output[i] << ", " << output[i + 1] << ", " << output[i + 2] << ", " << output[i + 3] << ") ";
        if (i % (width * 4) == 0) {
            std::cout << std::endl;
        }
    }
    
    // convert to 8 bit rgb image
    std::cout << "converting to 8 bit rgb image . . ." << std::endl;
    std::vector<unsigned char> output_x(width * height * 4);
    std::vector<unsigned char> output_y(width * height * 4);
    std::vector<unsigned char> output_full(width * height * 4);
    for (int i = 0; i < width * height * 4; i += 4) {
        output_x[i] = (unsigned char) (output[i] * 255);
        output_x[i + 1] = (unsigned char) (output[i] * 255);
        output_x[i + 2] = (unsigned char) (output[i] * 255);
        output_x[i + 3] = 255;

        output_y[i] = (unsigned char) (output[i + 1] * 255);
        output_y[i + 1] = (unsigned char) (output[i + 1] * 255);
        output_y[i + 2] = (unsigned char) (output[i + 1] * 255);
        output_y[i + 3] = 255;

        output_full[i] = (unsigned char) (output[i + 2] * 255);
        output_full[i + 1] = (unsigned char) (output[i + 2] * 255);
        output_full[i + 2] = (unsigned char) (output[i + 2] * 255);
        output_full[i + 3] = 255;
    }


    // write image as jpg
    std::cout << "saving image . . ." << std::endl;
    stbi_write_jpg("res/output/edge_x.jpg", width, height, 4, output_x.data(), 100);
    stbi_write_jpg("res/output/edge_y.jpg", width, height, 4, output_y.data(), 100);
    stbi_write_jpg("res/output/edge_full.jpg", width, height, 4, output_full.data(), 100);
    std::cout << "images saved" << std::endl;

    return 0;
}
