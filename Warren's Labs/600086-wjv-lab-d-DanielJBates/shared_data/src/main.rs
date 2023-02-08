mod shared_data;
use shared_data::SharedData;

fn main() {
    let data = SharedData::new();
    
    // Add thread creation code here, with processData as the thread main.
}

fn processData() {
    // Add code here to call update and print methods within SharedData
    // Note: You'll need to pass in SharedData as a parameter
}
