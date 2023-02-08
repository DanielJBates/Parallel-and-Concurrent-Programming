#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "ParticleSystem.h"

#include <stdio.h>
#include <chrono>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void dynamicsKernel(ParticleSystem* ps, float delta_t)
{
    int i = threadIdx.x;
    ps->particles[i].move_particle(delta_t);
}

int main()
{
    ParticleSystem ps;
    
    float current = std::chrono::system_clock::now();

    while (true)
    {
        for (int i = 0; i < 5; i++)
        {
            float x = -0.05 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.05 - -0.05)));
            float y = 0.975 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1.0 - 0.975)));
            float degree = -30.0 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (30.0 - -30.0)));
            float vx = tan(degree) * -1.0;

            Particle p = Particle(x, y, vx);

            ps.particles.push_back(p);
        }

        cudaError_t cudaStatus;

        ParticleSystem* gpu_ps;

        cudaStatus = cudaMalloc((void*)&gpu_ps, ps.particles.size() * sizeof(Particle));

        dynamicsKernel<<< 1, ps.particles.size()>>> ()
    }

    return 0;
}
