# 600086-Lab-D

## Ownership limitations

In the lecture and Lab C we looked at Rust class called `SharedData`.  This class is used for sharing of data across two or more threads

```Rust
pub struct SharedData {
    value: u32
}

impl SharedData {
    pub fn new() -> SharedData {
        SharedData { 
            value: 0 
        }
    }

    pub fn update(&mut self) {
        let local_value = self.value;
        std::thread::sleep(std::time::Duration::new(1,0));
        self.value = local_value + 1;
    }

    pub fn print(&self) {
        println!("SharedData: value = {}", self.value)
    }
}
```

In Lab C you created a new thread function which takes `SharedData` as a parameter and calls the `update` and `print` functions.

When you tried to move the `print` function from your thread function to the main program.  You would have experienced ownership issues.

These cannot be resolved by the ownership techniques we learnt earlier.  To solve this particular problem will require some new Rust techniques.  This lab begins by exploring one of these, namely: reference counters.

Create a new Rust folder and add the following code to main.rs

```Rust
struct Aircraft<'a> {
    name: String,
    engines: Vec<&'a Engine>,
}

impl Aircraft<'_> {
    pub fn new(name_param: &str) -> Aircraft {
        Aircraft {
            name: name_param.to_string(),
            engines: Vec::new()
        }
    }
}

struct Engine {
    name: String,
}

impl Engine {
    pub fn new(name_param: &str) -> Engine {
        Engine { 
            name: name_param.to_string(),
        }
    }
}

fn main() {
    let engine1 = Engine::new( "General Electric F404" );
    let engine2 = Engine::new( "General Electric F404" );
    let mut f18 = Aircraft::new( "F-18" );

    f18.engines.push (&engine1);
    f18.engines.push (&engine2);

    println! ("Aircraft: {} has a {} and {} ", f18.name, f18.engines[0].name, f18.engines[1].name );
}
```

This code creates two structs `Aircraft` and `Engine`.  `Aircraft` contains a vector of references that can hold a number of engines.

In `main` we create two `Engine` objects and an `Aircraft` object and the then link them together.

Examine the code and make sure you understand the syntax.

Note: The strange `'a` notation attached to the reference is called a lifetime parameter.  It allows the compiler to determine whether all references are going to stay "alive" at least as long as the "parent".  In our example, it ensure that the engines will exist at least as long as the aircraft.  What would happen if this was not the case?

Try to expand the code to include a data member in `Engine` that links to the `Aircraft`.

The limitation with the current code is that due to ownership restrictions it is not possible to link the `Aircraft` to an `Engine`

## Reference counters

Let's rewrite the code and use reference counters.

Create a new Rust folder and add the following code to main.rs

```Rust
use std::rc::Rc;

struct Aircraft {
    name: String,
    engines: Vec<Rc<Engine>>,
}

impl Aircraft {
    pub fn new(name_param: &str) -> Aircraft {
        Aircraft {
            name: name_param.to_string(),
            engines: Vec::new()
        }
    }
}

struct Engine {
    name: String,
}

impl Engine {
    pub fn new(name_param: &str) -> Engine {
        Engine {
            name: name_param.to_string(),
        }
    }
}

fn main() {
    let engine1 = Rc::new(Engine::new( "General Electric F404" ));
    let engine2 = Rc::new(Engine::new( "General Electric F404" ));

    let mut f18 = Aircraft::new( "F-18" );
    
    f18.engines.push (engine1.clone());
    f18.engines.push (engine2.clone());

    println! ("Aircraft: {} has a {} and {}", f18.name, f18.engines[0].name , f18.engines[1].name );
    println! ("Engine: {} ", engine1.name );
    println! ("Engine: {} ", engine2.name );
}
```

This code uses `RC` (or reference counters) to act as smart pointers to the objects.  We have removed the need to use references and lifetime parameters.  Arguable this code is also now easier to understand.

Examine the code.

Use your knowledge of the Rust ownership model to explain what is happening with the reference counters and why we do not need to pass them as references.

Remove the `clone()` method from this line

```Rust
f18.engines.push (engine1.clone());
```

Can you explain why this program now fails to build?

Add a new boolean data member `requires_service` to `Engine`.

Then add a new method `service(&mut self)` to `Engine`.  This method will just set the `requires_service` data member to `false`.

Now test your code with by adding the following to `main()`

```Rust
    let mut engine3 = Engine::new( "General Electric F404" );
    engine3.service();
```

You should be able to service engine3.

You will get a build error if you try and service engine2, which is accessed through an `rc`.  

Again, using your knowledge of the Rust ownership module can you explain why the error is occurring?

This is a limitation of reference counters.  We'll look to overcome this limitation in future labs.
