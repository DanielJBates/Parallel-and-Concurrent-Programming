const COOL_FACTOR: f32 = 0.75;


pub struct Particle 
{
    x: f32,
    y: f32,
    mass: f32,
    temperature: f32,
    velocity_x: f32,
    velocity_y: f32,
}

impl Particle 
{
    pub fn new(x_param: f32, y_param: f32, vx_param: f32) -> Particle 
    {
        Particle
        {
            x: x_param,
            y: y_param,
            velocity_x: vx_param,
            velocity_y: -1.0,
            mass: 1.0,
            temperature: 1.0,
        }
    }

    pub fn merge(p1: &Particle, p2: &Particle) -> Particle
    {
        Particle 
        {
            x: (p1.x + p2.x) / 2.0,
            y: (p1.y + p2.y) / 2.0,
            mass: p1.mass + p2.mass,
            temperature: (p1.temperature + p2.temperature) / 2.0,
            velocity_x: (p1.mass * p1.velocity_x + p2.mass * p2.velocity_x) / (p1.mass + p2.mass),
            velocity_y: -1.0,
        }
    }

    pub fn move_particle(&mut self, delta_t: f32)
    {   
        self.x = self.x + (self.velocity_x * delta_t);
        self.y = self.y + (self.velocity_y * delta_t);
    }

    pub fn cool_particle(&mut self, delta_t: f32)
    {
        let mut new_temp = self.temperature - delta_t * COOL_FACTOR / self.mass;

        if new_temp < 0.0
        {
            new_temp = 0.0;
        }

        self.temperature = new_temp;
    }

    pub fn wall_collision(&mut self)
    {
        if ((-0.5 + (0.01 * self.mass))) > self.x
        {
            self.x = -0.5 + (0.01 * self.mass);
            self.velocity_x = -self.velocity_x;
            return
        }
        if self.x > (0.5 - (0.01 * self.mass))
        {
            self.x = -0.5 + (0.01 * self.mass);
            self.velocity_x = -self.velocity_x;
            return
        }
    }

    pub fn floor_collision(&self) -> bool
    {
        if self.y > (-0.94 + (0.01 * self.mass))
        {
            return false
        }

        return true
    }

    pub fn get_x(&self) -> f32
    {
        return self.x
    }
    pub fn get_y(&self) -> f32
    {
        return self.y
    }
    pub fn get_temperature(&self) -> f32
    {
        return self.temperature
    }
    pub fn get_mass(&self) -> f32
    {
        return self.mass
    }
}

pub struct ParticleSystem 
{
    pub particles: Vec<Particle>
}

impl ParticleSystem {
    pub fn new() -> ParticleSystem
    {
        ParticleSystem 
        { 
            particles: Vec::new()
        }
    }

    pub fn particle_collision(p0: &Particle, p1: &Particle) -> bool
    {
        let c_squared = ((p1.get_x() - p0.get_x()) * (p1.get_x() - p0.get_x())) + ((p1.get_y() - p0.get_y()) * (p1.get_y() - p0.get_y()));
        let c = c_squared.sqrt();
    
        if c < ((0.01 * p0.get_mass()) + (0.01 * p1.get_mass()))
        {
            return true;
        }
    
        return false;
    }
}
