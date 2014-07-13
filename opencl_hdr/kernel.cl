

__kernel void separateChannels(__global unsigned char* input, __global unsigned char* bChannel,
                               __global unsigned char* gChannel, __global unsigned char* rChannel,            
                               const unsigned int width, const unsigned int height,
                               __constant float* filter, const int filter_width)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gSizeX = get_global_size(0);
    int i=0,j=0;
    
    if( (x * y) < (int) (width * height)) {
        bChannel[y * gSizeX + x] = 0;
        gChannel[y * gSizeX + x] = 0;
        rChannel[y * gSizeX + x] = 0;
        
        for(i = -filter_width/2; i <= filter_width/2; i++)
        {
            for(j = -filter_width/2; j <= filter_width/2; j++)
            {
                bChannel[y * gSizeX + x] += input[( (y + i) * gSizeX + x + j) * 3 + 0 ] * filter[(j+1) + (i+1) * 3];
                gChannel[y * gSizeX + x] += input[( (y + i) * gSizeX + x + j) * 3 + 1 ] * filter[(j+1) + (i+1) * 3];
                rChannel[y * gSizeX + x] += input[( (y + i) * gSizeX + x + j) * 3 + 2 ] * filter[(j+1) + (i+1) * 3];
            }
        }
    }
}

