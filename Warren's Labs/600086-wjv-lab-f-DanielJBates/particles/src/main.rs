extern crate rand;

const NUM_OF_THREADS: usize = 4;
const PARTICLES_PER_THREAD: usize = 25;

#[derive(Debug, Copy, Clone)]
struct Particle {
    x: f32,
    y: f32,
}
impl Particle {
    pub fn new(x_param: f32, y_param: f32) -> Particle {
        Particle {
            x: x_param,
            y: y_param,
        }
    }
}
struct ParticleSystem {
    particles: Vec<Particle>,
}
impl ParticleSystem {
    pub fn new() -> ParticleSystem {
        ParticleSystem {
            particles: vec![Particle::new(0.0, 0.0); NUM_OF_THREADS * PARTICLES_PER_THREAD],
        }
    }
    pub fn print_all(&self)
    {
        for i in 0..self.particles.len() {
            println!("Particle {} = {} , {}", i, self.particles[i].x, self.particles[i].y);
        }
    }
    pub fn move_particles(&mut self) 
    {
        for i in 0..self.particles.len() {
            let rx = rand::random::<f32>();
            let ry = rand::random::<f32>();

            self.particles[i].x = rx;
            self.particles[i].y = ry;
        }
    }
    pub fn move_particles_10_secs(&mut self)
    {
        let mut current = std::time::Instant::now();
        let mut last = current;
        let mut delta_time = current - last;

        loop {
            ParticleSystem::move_particles(self);
            println!("{} , {}", &self.particles[0].x, &self.particles[0].y);

            current = std::time::Instant::now();
            delta_time += current - last;
            last = current;

            if delta_time >= std::time::Duration::new(10,0) {
                break;
            }
        }   
    }
}

fn main() {
    let mut ps = ParticleSystem::new();

    ps.move_particles();
    //ps.move_particles_10_secs();

    ps.print_all();
}
