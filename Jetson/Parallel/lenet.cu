#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

__global__ void conv_valid (const double *input, double *output, const double *weight, int num)
{
	int IN_SIZE =0;
	int OUT_SIZE =0;
	int out_x = threadIdx.x;
	int out_y = threadIdx.y;
	if ( num == 1)
	{
		IN_SIZE = 32;
		OUT_SIZE = 28;
	}
	if(num ==2)
	{
		IN_SIZE = 14;
		OUT_SIZE = 10;

	}
	if(num ==3)
	{
		IN_SIZE = 5;
		OUT_SIZE =1;

	}

	if(out_x < OUT_SIZE && out_y < OUT_SIZE)
	{
		double sum =0.0f;

		for(int i=0; i < 5;i++)
		{
			for(int j=0; j<5;j++)
			{
				double val =input[(out_y+i) * IN_SIZE + (out_x+j)];
				double kernel = weight[i*5+j];
				sum += val*kernel;
			}
		}
		output[out_y * OUT_SIZE +out_x] += sum;
	}


}

__global__ void conv_bias (double *output, const double *bias, const int out_size)
{
	
	int c = blockIdx.x;
	int idx = threadIdx.x;

	output[c* out_size +idx] = fmax(0.0,output[c*out_size+idx] + bias[c]);

}

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}																						\

#define CONVOLUTION_FORWARD(input,output,weight,bias,action,num)				\
{																				\
	double *d_input, *d_output, *d_weight;										\
	int size_in = 0, size_ou = 0, size_we =0;									\
	int in_size = 0;															\
	int out_size =0;															\
	int output_channels=0;														\
	if (num ==1)																\
	{															 				\
		size_in = sizeof(double) *1* 32 *32; /**1 * 32 *32 ;*/					\
		size_ou = sizeof(double)*6* 28 *28; /*6 *28 *28*/;						\
		size_we = sizeof(double) *6*5*5;										\
		in_size = 32*32;														\
		out_size=28*28;															\
		output_channels = 6;													\
	}																			\
	if (num ==2)																\
	{															 				\
		size_in = sizeof(double) * 6* 14 * 14 ;									\
		size_ou = sizeof(double)* 16*10 *10;									\
		size_we = sizeof(double)*6*16*5*5;										\
		in_size = 14*14;														\
		out_size=10*10;															\
		output_channels = 16;													\
	}																			\
	if (num ==3)																\
	{															 				\
		size_in = sizeof(double)* 16*5 *5 ;										\
		size_ou = sizeof(double)* 120 * 1 *1;									\
		size_we = sizeof(double)*120*16*5*5;									\
		in_size = 5*5;															\
		out_size=1*1;															\
		output_channels = 120;													\
	}																			\
	double *input_flat = (double*)input;										\
	double *output_flat = (double*)output;										\
	double *weight_flat = (double*)weight;										\
	cudaMalloc((void**) &d_input, size_in);										\
	cudaMalloc((void**) &d_output, size_ou);									\
	cudaMalloc((void**) &d_weight, size_we);									\
	dim3 blockDim(32,32);														\
	dim3 gridDim(1,1);															\
	cudaMemcpy(d_input,input_flat,size_in,cudaMemcpyHostToDevice);				\
	cudaMemcpy(d_output,output_flat,size_ou,cudaMemcpyHostToDevice);			\
	cudaMemcpy(d_weight,weight_flat,size_we,cudaMemcpyHostToDevice);			\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
		{																		\
			int intput_offset =x*in_size;										\
			int output_offset =y*out_size;										\
			int weight_offset =(x*output_channels+y)*5*5;						\
			conv_valid<<<gridDim,blockDim>>>(d_input+intput_offset,d_output+output_offset,d_weight+weight_offset,num);	\
		}																		\
	cudaMemcpy(output, d_output, size_ou, cudaMemcpyDeviceToHost);				\
	double *output_flat2 = (double*)output;										\
	double *d_bias;																\
	double *d_output2;															\
	cudaMalloc((void**)&d_bias, output_channels*sizeof(double));				\
	cudaMalloc((void**)&d_output2, size_ou);									\
	cudaMemcpy(d_output2,output_flat2,size_ou,cudaMemcpyHostToDevice);			\
	cudaMemcpy(d_bias,bias,output_channels*sizeof(double),cudaMemcpyHostToDevice);\
	conv_bias<<<output_channels,out_size>>>(d_output2, d_bias, out_size);		\
	cudaMemcpy(output, d_output2,size_ou, cudaMemcpyHostToDevice);				\
	cudaFree(d_input);															\
	cudaFree(d_output);															\
	cudaFree(d_weight);															\
	cudaFree(d_bias);															\
	cudaFree(d_output2);														\
}																				\


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}


#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}


double relu(double x)
{
	return x*(x > 0);
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{

	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action,1);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action,2);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action,3);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}


static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}

}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
