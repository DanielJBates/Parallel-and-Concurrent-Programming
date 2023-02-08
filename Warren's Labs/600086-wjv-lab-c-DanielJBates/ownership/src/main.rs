
struct Person {
    name: String,
    age: u32
}

impl Person {
    // fn new_default() -> Person {
    //     Person {
    //         name: "Joe Bloggs".to_string(),
    //         age: 25
    //     }
    // }

    fn new(name_param: &str, age_param: u32) -> Person {
        Person {
            name: name_param.to_string(),
            age: age_param
        }
    }
}

fn main() {
    let p1 = Person::new("Jane", 30);

    print_person(p1);
}

fn print_person(p: Person) {
    println!("{} is {} years old", p.name, p.age);
}