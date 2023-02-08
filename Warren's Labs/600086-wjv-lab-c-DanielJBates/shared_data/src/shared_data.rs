
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
