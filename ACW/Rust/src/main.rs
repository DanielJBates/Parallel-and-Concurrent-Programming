#[macro_use]
extern crate glium;
extern crate rand;

use std::sync::{Arc, atomic::{AtomicU64}, Mutex};

use rand::Rng;

mod particles;

fn main() {
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let _counter = Arc::new(AtomicU64::new(0));

    let mut rng = rand::thread_rng();

    let mut ps = particles::ParticleSystem::new();

    //let ps_arc = Arc::new(Mutex::new(ps));

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
        color: [f32; 3],
    }

    implement_vertex!(Vertex, position, color);

    //Cubical
    let vertex0 = Vertex  {position: [-0.5, 1.0], color: [1.0, 1.0, 1.0]};
    let vertex1 = Vertex  {position: [0.5, 1.0], color: [1.0, 1.0, 1.0]};
    let vertex2 = Vertex  {position: [0.5, -1.0], color: [1.0, 1.0, 1.0]};
    let vertex3 = Vertex  {position: [-0.5, -1.0], color: [1.0, 1.0, 1.0]};

    let cubical = vec![vertex0, vertex1, vertex2, vertex3];

    let vertex_buffer_0 = glium::VertexBuffer::new(&display, &cubical).unwrap();
    let indices_0 = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[0u16,1,2,2,3,0]).unwrap();

    //Floor
    let vertex4 = Vertex {position:[-0.5, 0.03], color: [0.5, 0.5, 0.5]};
    let vertex5 = Vertex {position:[0.5, 0.03], color: [0.5, 0.5, 0.5]};
    let vertex6 = Vertex {position:[0.5, -0.03], color: [0.5, 0.5, 0.5]};
    let vertex7 = Vertex {position:[-0.5, -0.03], color: [0.5, 0.5, 0.5]};

    let floor = vec![vertex4, vertex5, vertex6, vertex7];

    let vertex_buffer_1 = glium::VertexBuffer::new(&display, &floor).unwrap();
    let indices_1 = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[0u16,1,2,2,3,0]).unwrap();

    //Shower Head
    let vertex8 = Vertex {position: [-0.05, 0.025], color: [0.5, 0.5, 0.5]};
    let vertex9 = Vertex {position: [0.05, 0.025], color: [0.5, 0.5, 0.5]};
    let vertex10 = Vertex {position: [0.05, -0.025], color: [0.5, 0.5, 0.5]};
    let vertex11 = Vertex {position: [-0.05, -0.025], color: [0.5, 0.5, 0.5]};

    let shower_head = vec![vertex8, vertex9, vertex10, vertex11];

    let vertex_buffer_2 = glium::VertexBuffer::new(&display, &shower_head).unwrap();
    let indices_2 = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[0u16,1,2,2,3,0]).unwrap();

    let vertex_shader_src = r#"
        #version 140

        in vec2 position;
        in vec3 color;

        out vec3 vColor;

        uniform mat4 matrix;

        void main() {
            gl_Position = matrix * vec4(position, 0.0, 1.0);
            vColor = color;
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        in vec3 vColor;

        out vec4 fColor;

        void main() {
            fColor = vec4(vColor, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut _current = std::time::Instant::now();
    let mut _last = _current;
    let mut _delta_t = _last - _current;

    event_loop.run(move |event, _, control_flow| {

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        // Begin render loop

        _current = std::time::Instant::now();
        _delta_t = _current - _last;
        _last = _current;

        // Create a drawing target
        let mut target = display.draw();

        // Clear the screen to black
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        let _ptr = program.get_frag_data_location("fColor").unwrap();

        //Draw Cubical Start
        let pos_x : f32 = 0.0;
        let pos_y : f32 = 0.0;
        let pos_z : f32 = 0.0;

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [pos_x, pos_y, pos_z, 1.0],
            ]
        };

        target.draw(&vertex_buffer_0, &indices_0, &program, &uniforms, &Default::default()).unwrap();
        //Draw Cubical End

        //Draw Floor Start
        let pos_x : f32 = 0.0;
        let pos_y : f32 = -0.97;
        let pos_z : f32 = 0.0;

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [pos_x, pos_y, pos_z, 1.0],
            ]
        };

        target.draw(&vertex_buffer_1, &indices_1, &program, &uniforms, &Default::default()).unwrap();
        //Draw Floor End

        //Draw Particles here

        let mut dynamics_pool = scoped_threadpool::Pool::new(2);

        dynamics_pool.scoped(|scope| {

            for _i in 0..5 {

                let x: f32 = rng.gen_range(-0.05, 0.05);
                let y: f32 = rng.gen_range(0.975, 1.0);
                let degree: f32 = rng.gen_range(-30.0, 30.0);

                let vx: f32 = degree.tan() * -1.0;

                let particle = particles::Particle::new(x,y,vx);

                ps.particles.push(particle);
            }

            let particles_per_thread = ps.particles.len()/2;
            for chunk in ps.particles.chunks_mut(particles_per_thread) {
                scope.execute(move || thread_dynamics(chunk, _delta_t.as_secs_f32()));
            }
        });

        let mut temperatures_pool = scoped_threadpool::Pool::new(2);

        temperatures_pool.scoped(|scope| {
            let particles_per_thread = ps.particles.len()/2;
            for chunk in ps.particles.chunks_mut(particles_per_thread) {
                scope.execute(move || thread_temperature(chunk, _delta_t.as_secs_f32()));
            }
        });

        /*let mut collisions_pool = vec!();

        let wall_thread_ps = ps_arc.clone();
        let wall_thread = std::thread::spawn(move || thread_wall_collision(wall_thread_ps));
        collisions_pool.push(wall_thread);

        let floor_thread_ps = ps_arc.clone();
        let mut counter_clone = counter.clone();
        let floor_thread = std::thread::spawn(move || thread_floor_collision(floor_thread_ps, &mut counter_clone));
        collisions_pool.push(floor_thread);

        let particles_thread_ps = ps_arc.clone();
        let particles_thread = std::thread::spawn(move || thread_wall_collision(particles_thread_ps));
        collisions_pool.push(particles_thread);

        for t in collisions_pool {
            let _result = t.join();
        }
        */

        /*let draw_ps = ps_arc.clone();
        let draw_guard = draw_ps.lock().unwrap();

        let num_of_particles = draw_guard.particles.len();*/
        let num_of_particles = ps.particles.len();
        for i in 0..num_of_particles {

            let temperature = ps.particles[i].get_temperature();
        let vertex12 = Vertex {position: [-0.01, 0.01], color: [(0.0 + temperature), 0.0, (1.0 - temperature)]};
        let vertex13 = Vertex {position: [0.01, 0.01], color: [(0.0 + temperature), 0.0, (1.0 - temperature)]};
        let vertex14 = Vertex {position: [0.01, -0.01], color: [(0.0 + temperature), 0.0, (1.0 - temperature)]};
        let vertex15 = Vertex {position: [-0.01, -0.01], color: [(0.0 + temperature), 0.0, (1.0 - temperature)]};

        let particle_square = vec![vertex12, vertex13, vertex14, vertex15];

        let vertex_buffer_3 = glium::VertexBuffer::new(&display, &particle_square).unwrap();
        let indices_3 = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &[0u16,1,2,2,3,0]).unwrap();

        let pos_x = ps.particles[i].get_x();
        let pos_y  = ps.particles[i].get_y();
        let pos_z = 0.0;
        let scale = ps.particles[i].get_mass();

        let uniforms = uniform! {
            matrix: [
                [scale, 0.0, 0.0, 0.0],
                [0.0, scale, 0.0, 0.0],
                [0.0, 0.0, scale, 0.0],
                [pos_x, pos_y, pos_z, 1.0],
            ]
        };

        target.draw(&vertex_buffer_3, &indices_3, &program, &uniforms, &Default::default()).unwrap();
    }

        //Draw Shower Head Start
        let pos_x : f32 = 0.0;
        let pos_y : f32 = 0.975;
        let pos_z : f32 = 0.0;

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [pos_x, pos_y, pos_z, 1.0],
            ]
        };

        target.draw(&vertex_buffer_2, &indices_2, &program, &uniforms, &Default::default()).unwrap();
        //Draw Shower Head End

        // Display the completed drawing
        target.finish().unwrap();

        // End render loop
    });
}

fn thread_dynamics(chunk: &mut [particles::Particle], delta_t: f32)
{
    for i in 0..chunk.len() {
        chunk[i].move_particle(delta_t)
    }
}

fn thread_temperature(chunk: &mut [particles::Particle], delta_t: f32)
{
    for i in 0..chunk.len() {
        chunk[i].cool_particle(delta_t)
    }
}

fn thread_wall_collision(ps: Arc<Mutex<particles::ParticleSystem>>)
{
    let mut guard = ps.lock().unwrap();

    for i in 0..guard.particles.len() {
        guard.particles[i].wall_collision();
    }
}

fn thread_floor_collision(ps: Arc<Mutex<particles::ParticleSystem>>, counter: &mut Arc<AtomicU64>)
{
    let mut ps_guard = ps.lock().unwrap();

    for i in 0..ps_guard.particles.len() {
       if (ps_guard.particles[i].floor_collision())
       {
           counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
           ps_guard.particles.remove(i);
       }
    }
    println!("{}", counter.load(std::sync::atomic::Ordering::Relaxed));
}

fn thread_particle_collision(ps: Arc<Mutex<particles::ParticleSystem>>)
{
    let mut ps_guard = ps.lock().unwrap();

    for i in 0..ps_guard.particles.len() {
        for j in 0..ps_guard.particles.len() {
            if (i == j) {
                continue;
            }
            if(particles::ParticleSystem::particle_collision(&ps_guard.particles[i], &ps_guard.particles[j]))
            {
                let new_particle = particles::Particle::merge(&ps_guard.particles[i], &ps_guard.particles[j]);
                ps_guard.particles.push(new_particle);
            }
        }
    }
}