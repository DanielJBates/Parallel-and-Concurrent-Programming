# A shower simulation, in parallel

## Concept

Implement two versions of a shower simulation, one on the CPU and the other on the GPU.

## Specification

The scene is composed of a shower cubical with height 2m, width 1m and depth 1m.

The water comes from a small circular area (the shower head), of diameter 0.1m, located in the centre of the cubical roof.

Each water molecule is represented by a separate particle in a particle system

### Particle dynamics

Each water molecule (particle) is spawned from a random position in the shower head. It then exits the shower heard in a random direction between -30 to +30 degrees to vertically down i.e. the shower head sprays water in a conical downward pattern.  The initial speed of all particles is defined by a single global variable initial_speed (see below)

All particles are subject to gravity.

A particles is removed from the simulation when it hits the shower floor, and a new particle is created at the shower head.  Thus the simulation has a fixed number of active particles (except if there are collisions, see below).

The program needs to maintain a count of the total number of particles that hit the shower floor.

If a particle hits a side of the cubical, then it can either bounce back into the cubical or be placed at the opposite side of the cubical (i.e. as though space has been wrapped).  The choice of implementation is left open.

### Particle collisions

All particles have a starting mass of 1

Particles can collide with other particles.  

Upon a collision, the particles merge to form a single new particle with the combined mass of the two original particles.  

The velocity of the new particle is defined as:

`v = (m0 * v0 + m1 * v1) / (m0 + m1)`

where

- `v0` and `v1` are the velocities of two colliding particles
- `m0` and `m1` are the masses of the two colliding particles

### Particle temperature

All particles have a starting temperature of 1, and gradually cool down as they fall.

Particle temperature can range from 1 (hot) to 0 (cold).

Particles cool down is defined by:

`temp1 = temp0 – deltaT * cool_factor / mass`

where

- `temp0` is the current temperature
- `temp1` is the  new temperature
- `deltaT` is the time since last temperature calculation
- `mass` is the mass of the particle
- `cool_factor` is a global variable to control the cooling rate of particles (see later)

The larger the particle the slower it cools.

### Parallel architecture

There are specific requirements for the use of threads, as specified below:

CPU implementation contains at least 6 threads:

- Particle dynamics - implemented on at least 2 additional threads
- Particle collisions - implemented on at least 2 additional threads
- Particle temperature - implemented on at least 2 additional threads

GPU implementation contains at least `2N+1` threads (where `N` is the number of particles):

- Particle dynamics - implemented as 1 thread per particle
- Particle collisions – implemented on 1 or more threads
- Particle temperature - implemented as 1 thread per particle

### Visualization

All particles are to be rendered using OpenGL, as discussed in previous labs.  Each particle is to be rendered as a triangle or sphere (or more complex shape).

Two graphical modes are required, with switching via a key

1. Temperature: The particle colour is selected based on a sliding scale where `temperature = 0` is blue, and `temperature = 1` is red.
2. Mass: The particle colour is select based on its mass. The choice of colour is left open.

### Tuning

You are required to tune the `cool_factor` and `initial_speed` global variables to achieve the following results.

`Initial_speed` adjusted so particles exit the show head with sufficient velocity so that only a small number of particles collide with the cubical walls.

`Cool_factor` adjusted so that the majority of the smaller particles are cold by the time they reach the floor.

### Performance mode

A key part of this assessment is to investigate the performance characteristics of your simulation(s).

Increase the number of particles within your system until your simulation is no longer capable of maintaining smoothly moving particles.

Whilst your simulation should be capable of running with 100k+ particles, the graphics component will not.  To compensate, alter your graphics so that you only visualise 1 in every N particles.  E.g. render 1 particle in every 100.  This allows you to still visualise the simulation without having to suffer a low frame rate, due to a limitation of the graphics.

## Implementation

### CPU

Rust implementation

### GPU

The choice of language from which to host CUDA is left open e.g. C++, C# or Python

## Deliverables

### Rust (CPU) implementation (Code)

Specified above

### CUDA (GPU) implementation (Code)

Specified above.

### Lab book entries

1. GPU design - Detail the design of your GPU implementation.  The focus of the report is on the parallel aspects of your solution, so clearly explain how you have used threads, mutual exclusion and synchronization in your GPU implementation.  (approx. 500 words)
2. CPU design - Detail the design of your CPU implementation.  The focus of the report is on the parallel aspects of your solution, so clearly explain how you have used threads, mutual exclusion and synchronization in your CPU implementation.  (approx. 500 words)
3. Performance comparison - Compare the performance of your GPU and CPU implementations.  Include a description of your performance benchmarks, together with a comparison of performances between different ways of partitioning the tasks and thread blocks.  (approx. 500 words)
4. Reflection - What have you learnt about parallel software development?  What aspects of your solutions worked well and which need further refinement? In hindsight, is there anything you would have done differently?  (approx. 500 words)

### Video

A short narrated video containing showing both the CPU and GPU implementations

### Demonstration

You will be required to demonstrate your ACW on windows 10 PC running CUDA

## Submission

All submissions are via Canvas.

### 1. Software

Your code, in the form of a Visual Studio solution, source code, and working executables/DLLs, along with any required assets.

### 2. Lab book

The full lab book containing all labs.

### 3. Video
