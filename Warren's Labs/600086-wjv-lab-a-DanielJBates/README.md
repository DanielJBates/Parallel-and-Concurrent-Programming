# 600086-Lab-A

This lab takes you through the process of installing and setting up your Rust environment, and gets you started on learning Rust.

There is a video on Canvas to accompany this setup procedure.

## Installation

Download and install Visual Studio Code, from: https://code.visualstudio.com/

Download and install Rust, from https://rustup.rs/

Open VS Code and select the **Extensions Manager** on the left hand side of the interface.

Search with the keyword "Rust"

Install "rust-analyzer" by matklad

Search with the keyword "TOML"

Intall "TOML Language Support" by be5invis

## Q1. Hello World

Create a new project folder `hello_world` in a suitable location.

Open a **Terminal Window** by selecting **Terminal->New Terminal**

Within the new terminal window, use the command `cd` to change to the location of your newly created `hello_world` folder.

Example:

```dos
cd "C:\Users\Warren\Documents\Visual Studio Projects\GitHub\600086-Lab\hello_world"
```

Note: if you have spaces in the path, then you'll need to enclose the path within ""

Now type the following in the **Terminal Window**

```dos
cargo init
```

This setups up your hello_world project.  By default this also adds the "hello World" print statement to a single rust file.

Open the new folder using **File->Open Folder**.

Compile and run the code by typing the following into the Terminal Window:

```dos
cargo run
```

Congratualtions, you have written your first Rust program.

## Q2. Learning Rust

We'll use Rust within this module as a vehicle to learn and practice parallel programming.  To this end its important that you become familiar with the Rust language.  

There are a huge number of online resources to assist you in learning Rust.  One that I can recommend is: https://www.youtube.com/watch?v=zF34dRivLOw

Another useful resource is The Rust Programming Language available at https://doc.rust-lang.org/stable/book/ 

Work through this or a similar resource to become familiar with Rust.
