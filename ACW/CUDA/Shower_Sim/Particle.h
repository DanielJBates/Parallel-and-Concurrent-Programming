#pragma once
class Particle
{
public:
	float x;
	float y;
	float velocity_x;
	float velocity_y;
	float mass;
	float temperature;

	Particle(float x_param, float y_param, float vx_param);
	Particle(float x_param, float y_param, float vx_param, float mass_param, float temperature_param);
	Particle merge(Particle p0, Particle p1);
	void move_particle(float delta_t);
	void cool_particle(float delta_t);
	void wall_collision();
	bool floor_collision();
};

