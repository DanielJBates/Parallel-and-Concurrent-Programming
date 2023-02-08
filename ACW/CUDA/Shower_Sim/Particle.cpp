#include "Particle.h"

const float COOL_FACTOR = 0.75;

Particle::Particle(float x_param, float y_param, float vx_param)
{
	Particle::x = x_param;
	Particle::y = y_param;
	Particle::velocity_x = vx_param;
	Particle::velocity_y = -1.0;
	Particle::mass = 1.0;
	Particle::temperature = 1.0;
}

Particle::Particle(float x_param, float y_param, float vx_param, float mass_param, float temperature_param)
{
	Particle::x = x_param;
	Particle::y = y_param;
	Particle::velocity_x = vx_param;
	Particle::velocity_y = -1.0;
	Particle::mass = mass_param;
	Particle::temperature = temperature_param;
}

Particle Particle::merge(Particle p0, Particle p1)
{
	float x = (p0.x + p1.x) / 2.0;
	float y = (p0.x + p1.x) / 2.0;
	float vx = (p0.mass * p0.velocity_x + p1.mass * p1.velocity_x) / (p0.mass + p1.mass);
	float mass = p0.mass + p1.mass;
	float temperature = (p0.temperature + p1.temperature) / 2.0;

	return Particle::Particle(x, y, vx, mass, temperature);
}

void Particle::move_particle(float delta_t)
{
	Particle::x = Particle::x + (Particle::velocity_x * delta_t);
	Particle::y = Particle::y + (Particle::velocity_y * delta_t);
}

void Particle::cool_particle(float delta_t)
{
	float new_temp = Particle::temperature - delta_t * COOL_FACTOR / Particle::mass;

	if (new_temp < 0.0)
	{
		new_temp = 0.0;
	}

	Particle::temperature = new_temp;
}

void Particle::wall_collision()
{
	if (((-0.5 + (0.01 * Particle::mass))) > Particle::x)
	{
		Particle::x = -0.5 + (0.01 * Particle::mass);
		Particle::velocity_x = -Particle::velocity_x;
		return;
	}
	if (Particle::x > (0.5 - (0.01 * Particle::mass)))
	{
		Particle::x = -0.5 + (0.01 * Particle::mass);
		Particle::velocity_x = -Particle::velocity_x;
		return;
	}
}

bool Particle::floor_collision()
{
	if (Particle::y > (-0.94 + (0.01 * Particle::mass)))
	{
		return false;
	}

	return true;
}