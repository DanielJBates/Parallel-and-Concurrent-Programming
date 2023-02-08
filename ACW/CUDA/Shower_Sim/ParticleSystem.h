#pragma once
#include <vector>;
#include "Particle.h"

class ParticleSystem
{
public:
	std::vector<Particle> particles;

	ParticleSystem();
	ParticleSystem(std::vector<Particle> vec_param);
	bool particle_collision(Particle p0, Particle p1);
};

