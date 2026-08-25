class TinyEnv:
    def __init__(self):
        self.state = 0
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self,action):
        
        if action == 1:
            reward = 1
        else :
            reward =0
        
        done = True
        next_state = 0
        return next_state, reward, done