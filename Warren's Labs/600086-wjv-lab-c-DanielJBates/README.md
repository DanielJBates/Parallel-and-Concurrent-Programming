# 600086-Lab-C

## Multiple Rust files

Multiple Rust files can be linked together into a single application with the use of the `mod` statement

Placing `mod my_second_file` in one Rust file, enables that file to use the functions from `my_second_file.rs`

Make use of this functionality to move your thread main function, from the previous lab to its own Rust file.  This method requires that you prefix any imported functions with `my_second_file`

```Rust
    my_second_file::run();
```

An alternative method is the `use` command which allows you to drop the requirement to prefix

```Rust
    mod my_second_file;
    use my_second_file::run;
```

## Ownership

Open up the folder **Ownership**.

Familiarise yourself with the `struct Person` and the `impl Person`.  The `struct` defines the data members for a Person.  The `impl` provides the behaviour for a Person.

```Rust
struct Person {
    name: String,
    age: u32
}
```

```Rust
impl Person {
    fn new_default() -> Person {
        Person {
            name: "Joe Bloggs".to_string(),
            age: 25
        }
    }

    fn new(name_param: &str, age_param: u32) -> Person {
        Person {
            name: name_param.to_string(),
            age: age_param
        }
    }
}
```

Note that Rust does not allow overriding of function names, so we have to define unique names for each "constructor".

A basic print function `print_person` has been provided to print out a person's details (this would ordinarily have been included in the `impl` section, but for simplicity its define outside as a basic function)

The main function, creates a `Person` struct and passes it to the `print_person`.

Add a second call to `print_person` within `main`, to print out the details a second time.  Why does this not compile?

Hint: Rust has very struct ownership rules.

Now alter the code so that the `print_person` returns the Person object back to the `main`.

It would be very cumbersome to have to pass objects into functions and then to return them, so ownership could be retained.  Rust has an alternative, the reference.

Modify `print_person` to use a reference

```Rust
    fn print_person(p: &Person)
```

Rust permits permits shared ownership on an immutable reference.  This is called **borrowing**.

You should now be able to use multiple calls to `print_person` without any issue.

Add the function `increment_age` which takes a mutable reference as a parameter.

```Rust
fn increment_age(p: &mut Person) {
    p.age = p.age + 1;
}
```

Call this function once then twice within `main`.  You'll also need to make the `Person` object mutable

```Rust
fn main() {
    let mut p1 = Person::new("Jane", 30);

    print_person(&p1);
    print_person(&p1);

    increment_age(&p1);
    increment_age(&p1);
}

fn print_person(p: & Person) {
    println!("{} is {} years old", p.name, p.age);
}

fn increment_age(p: &mut Person) {
    p.age = p.age + 1;
}
```

Rust has a rule that states you cannot have more than one mutable reference to the same object, neither can you have even a single mutable reference to an object that has one or more immutable references.

So why does the code you have created, work?

The answer is that Rust is able to determine if a reference (whether mutable or immutable) is still being used.

Now lets see if we can cause the borrowing to fail.  We'll add some explicit immutable and mutable references to our code.

```Rust
fn main() {
    let mut p1 = Person::new("Jane", 30);

    let r1 = & p1;
    let r2 = & p1;

    print_person(r1);
    print_person(r2);

    let r3 = &mut p1;

    increment_age(r3);
}
```

`r1`, `r2` and `r3` are references

This should compile and execute just fine.

Now move one of the calls of `print_person` below the call to `increment_age`.

The code should no longer compile.

Rust has seen that we are trying to use a mutable reference to an object for which there is still an active immutable reference.

You can find more details here: <https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html>

## Classes

In the lecture you were introduced to a Rust class called `SharedData`.  This class is used for sharing of data across two or more threads

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

Familiarise yourself with how Rust implements:

- the constructor i.e. `new`
- methods
- data members
- the self object (The equivalent to C#'s `this` keyword)

The main program illustrates how to use the `SharedData` struct

Notice how placing `SharedData` in a separate Rust file, hides all the components and we have to use the `pub` keyword to make them visible to `main`

Create a new thread function which takes `SharedData` as a parameter and then calls the `update` and `print` functions.

Now run your new thread function in a thread, as per the previous Lab

Once this is working, try and move the `print` function from your thread function to the main program.  You will notice that this causes ownership issues.

These cannot be resolved by the ownership techniques we learnt earlier.  To solve this particular problem will require some new Rust techniques e.g. `arc` or the asynchronous reference counters.  We'll investigate this in a future lab.

However, the reason why Rust does not allow the sharing of SharedData between main and your thread function is too eliminate the possibility of a race condition.  The Rust language was created in response to a requirement for a very efficient language that was safe to use for multi-threaded problems.
