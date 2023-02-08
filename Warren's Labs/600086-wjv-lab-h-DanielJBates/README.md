# 600086-Lab-H

In this lab we're going to expand on the synchronization examples covered in the lectures, and also explore the efficiency of one of the parallel patterns -- striped arrays -- in more detail.

## Condition variables

Create a producer / consumer architecture using condition variables.

A simple consumer / producer system was introduced during the lecture.

Implement this simple system in Rust

Now expand the example to work with multiple consumers and multiple producers.

Assess the scalability of such an approach.  Consider where you believe the bottlenecks to be, and how you could improve the overall performance.

## Striped arrays, sequential access

Open the example program `striped_array`.  This is very similar to the example code introduced in the lectures.  Examine the code, so that you understand how it is implemented.

Add timing code to the example to measure the duration of array access.  Be careful to avoid timing the print statements.

Also remember to use **release** mode when timing your code (i.e. `cargo run --release`).  Timing in standard (debug) mode is pointless, due to the amount of debug code injected automatically by the compiler.

You may find this useful

```Rust
 let start = SystemTime::now();
 ...
 start.elapsed().unwrap().as_micros();
 ```

Alter the program to use 2, 4, 8, 16, 32 threads.  As the number of threads increases you'll need to half the size of the strips i.e. 16384, 8192, 4096, 2048, ....

What do you notice about increasing the number of threads?

## Striped arrays, random access

Our implementation of striped arrays suffers in terms of performance when we add more threads.  This is due to the partly to the way the test code has been implemented i.e it uses sequential access.  

Replace

```Rust
data.write(j, id);
```

with

```Rust
let index = rand::random::<usize>() % data.length_of_strip*data.num_of_strips;
data.write(index, id);
```

This code now simulates random access to the our array.

Re-run your performance tests.  What are the results ?

## Striped arrays, finer grain locking (OPTIONAL)

The current implementation of our striped array uses one stripe per thread.  

Alter the code so that we can have N stripes per thread

Set N = 2, 4, 8 and rerun your tests.  Is there a noticeable difference ?

## Striped arrays, improved (OPTIONAL)

Whilst the striped implementation is good at allowing random access to shared data, it is certainly not optimal in allowing sequential access to the array.

You will have noticed there are lots of locks and unlocks, in our previous simulations.  Each write involves calling a lock.  This is very fine grain locking.  If we know we are going to update a section of the array, then an optimization would be to lock the array, perform all the writes and then unlock it.  Thereby saving many locks/unlocks.

Implement a new write function with the following interface and algorithm:

```Rust
fn write_strip (&self, strip_index: usize, value: usize)
```

1. lock the strip
2. write `value` to every element in the strip
3. unlock the strip

Rerun your timing tests using this new function.  Is there a noticeable difference ?
