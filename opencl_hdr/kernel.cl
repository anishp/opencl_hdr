

__kernel void separateChannels(__global unsigned char* input, __global unsigned char* bChannel,
                               __global unsigned char* gChannel, __global unsigned char* rChannel,            
                               const unsigned int width, const unsigned int height, __constant float* filter) 
 {                                                                                                            
   int i = get_global_id(0);                                                                                  
   if(i<(width*height)) {                                                                                     
     bChannel[i] = input[i * 3];                                                                              
     gChannel[i] = input[i * 3 + 1];                                                                          
     rChannel[i] = input[i * 3 +2]; }                                                                         
}