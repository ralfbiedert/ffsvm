use std::time::Instant;

pub struct ProfilerTrace {
    start: Instant,
    
    pub performance_numbers: Vec<u32>
}


impl ProfilerTrace {
    
    pub fn new() -> ProfilerTrace {
        
        ProfilerTrace {
            start: Instant::now(),
            performance_numbers: Vec::with_capacity(2048)
        }
    }
    
    pub fn restart(&mut self) {
        self.start = Instant::now();
    }
    
    
    pub fn snapshot(&mut self, id: u32) {
        let elapsed = Instant::now() - self.start;
        
        self.performance_numbers.push(id);
        self.performance_numbers.push(elapsed.subsec_nanos());
    }
    
    
}