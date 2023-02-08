#include <math.h>
#include "ParticleSystem.h"

ParticleSystem::ParticleSystem()
{
}

ParticleSystem::ParticleSystem(std::vector<Particle> vec_param)
{
	ParticleSystem::particles = vec_param;
}

bool ParticleSystem::particle_collision(Particle p0, Particle p1)
{
    float c_squared = ((p1.x - p0.x) * (p1.x - p0.x)) + ((p1.y - p0.y) * (p1.y - p0.y));
    float c = sqrt(c_squared);

    if (c < ((0.01 * p0.mass) + (0.01 * p1.mass)))
    {
        return true;
    }

    return false;
}
