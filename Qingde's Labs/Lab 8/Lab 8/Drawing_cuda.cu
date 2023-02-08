/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable.h"
#include "hitable_list.h"

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

//__device__ static int ticks = 1;
__device__ int timestep = 0.1;
static vec3 position = vec3(-1.5, 0, 0);
__device__ vec3 v = vec3(0.1, 0, 0);


cudaArray* d_imageArray = 0;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);

	}
}

__device__ vec3 castRay(const ray& r, hitable** world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		vec3 colour = 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
		colour = 0.5 * rec.colour + 0.5 * colour;
		return colour;
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void create_world(hitable** d_list, hitable** d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		
		//vec3 v = vec3(0.01, 0.0, 0);


		//if (v.x() >= 2.0 || v.x() <= -2.0)
		//{
		//	position += vec3(v.x() * timestep,0,0);
		//	*(d_list) = new sphere(vec3(position.x() - v.x(), 0 + v.y(), -0.5 + v.z()), 0.5);
		//	
		//	//v = vec3(-0.01 * (float)ticks++, 0.0 * (float)ticks++, 0);
		//}

		//if (sqrt(((-10002.0 - (-1.5 + v.x())) * (-10002.0 - (-1.5 + v.x()))) + ((0 - (0 + v.y())) * (0 - (0 + v.y())))) <= 10000 + 0.5
		//	||
		//	sqrt(((10002.0 - (-1.5 + v.x())) * (10002.0 - (-1.5 + v.x()))) + ((0 - (0 + v.y())) * (0 - (0 + v.y())))) <= 10000 + 0.5) //refelect v on x axis
		//{
		//	v = vec3(-(v.x()), v.y(), v.z());
		//}

		//if (sqrt(((0 - (-1.5 + v.x())) * (0 - (-1.5 + v.x()))) + ((10002.0 - (0 + v.y())) * (10002.0 - (0 + v.y())))) <= 10000 + 0.5
		//	||
		//	sqrt(((0 - (-1.5 + v.x())) * (0 - (-1.5 + v.x()))) + ((-10002.0 - (0 + v.y())) * (-10002.0 - (0 + v.y())))) <= 10000 + 0.5) //refelect v on y axis
		//{
		//	v = vec3(v.x(), -(v.y()), v.z());
		//}


		*(d_list) = new sphere(vec3(position.x() + v.x(), position.y() + v.y(), position.z() + v.z()), 0.5);
		//*(d_list) = new sphere(vec3(cos(0.01 * (float)ticks++), sin(0.01 * (float)ticks++), -0.5), 0.5);
		*(d_list + 1) = new sphere(vec3(-10002.0, 0, -3), 10000); //left
		*(d_list + 2) = new sphere(vec3(10002.0, 0, -3), 10000); //right
		*(d_list + 3) = new sphere(vec3(0, 10002.0, -3), 10000); //top
		*(d_list + 4) = new sphere(vec3(0, -10002.0, -3), 10000); //bottom
		*(d_list + 5) = new sphere(vec3(0, 0, -10000.5), 10000); //back

		*d_world = new hitable_list(d_list, 6);
	}
}
__global__ void free_world(hitable** d_list, hitable** d_world) {
	delete* (d_list);
	delete* (d_list + 1);
	delete* (d_list + 2); 
	delete* (d_list + 3); 
	delete* (d_list + 4);
	delete* (d_list + 5);
	delete* d_world;
}

__global__ void d_render(uchar4* d_output, uint width, uint height, hitable** d_world)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	float u = x / (float)width; //----> [0, 1]x[0, 1]
	float v = y / (float)height;

	u = 2.0 * u - 1.0; //---> [-1, 1]x[-1, 1]
	v = -(2.0 * v - 1.0);
	u *= width / (float)height;

	u *= 2.0;
	v *= 2.0;

	vec3 eye = vec3(0, 0.5, 1.5);
	float distFrEye2Img = 1.0;;
	if ((x < width) && (y < height))
	{
		//for each pixel
		vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
		//fire a ray:
		ray r;
		r.Origin = eye;
		r.Direction = pixelPos - eye; //view direction along negtive z-axis!
		vec3 col = castRay(r, d_world);
		float red = col.x();
		float green = col.y();
		float blue = col.z();
		d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
	}
}


	extern "C" void freeTexture() {

		checkCudaErrors(cudaFreeArray(d_imageArray));
	}

	// render image using CUDA
	extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 * output)
	{
		// make our world of hitables
		hitable** d_list;
		checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
		hitable** d_world;
		checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
		create_world << <1, 1 >> > (d_list, d_world);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		d_render << <gridSize, blockSize >> > (output, width, height, d_world);
		getLastCudaError("kernel failed");
	}
#endif