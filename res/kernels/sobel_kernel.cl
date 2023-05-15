__kernel void convolution(
    __read_only image2d_t input,
    __write_only image2d_t output,
    sampler_t sampler,
    __constant float *sobel_x,
    __constant float *sobel_y,
    int x_or_y,
    int filter_width
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int half_filter_width = filter_width / 2;

    int2 coord = {x, y};
    float4 result = {0, 0, 0, 0};
    int sobel_idx = 0;

    int2 local_pos = {0, 0};
    for (int i = -half_filter_width; i <= half_filter_width; i++) {
        local_pos.y = y + i;
        for (int j = -half_filter_width; j <= half_filter_width; j++) {
            local_pos.x = x + j;

            float4 pixel = read_imagef(input, sampler, local_pos);
            float greyscale = (pixel.x + pixel.y + pixel.z) / 3.0f;

            result.x += greyscale * sobel_x[sobel_idx];
            result.y += greyscale * sobel_y[sobel_idx];
            sobel_idx += 1;
        }
    }
    result.x = fabs(result.x);
    result.y = fabs(result.y);
    result.z = sqrt(pow(result.x, 2) + pow(result.y, 2));
    result.w = 1.0f;
    write_imagef(output, coord, result);
}