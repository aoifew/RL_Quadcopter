import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 3
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        #self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0.5*self.sim.v[2] #Initial reward - fraction of z-velocity (vertical direction)
        
        reward -= 0.5*abs(self.target_pos[2]-self.sim.pose[2]) #Penalise large vertical distance from the target
     
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 1.0 #BONUS:if the z-position has exceeded the target z-position
        
        if self.sim.pose[2] < self.sim.init_pose[2]:
            reward -= 1.0 #PENALTY: if the copter is below the starting point
        
        #BONUS: reward if the copter is getting close to the target
        #if abs(self.sim.pose[2] - self.target_pos[2]) <= 3: 
        #    reward += 1.0
        #else:
        #    reward -= 1.0
        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        rotor_speeds = np.array([rotor_speeds]*4)
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.target_pos[2]-self.sim.pose[2])
            pose_all.append(self.sim.pose[2])
            pose_all.append(self.sim.v[2])
            if self.sim.pose[2] >= self.target_pos[2]:
                done=True
        next_state = np.array(pose_all)
        return next_state, np.tanh(reward), done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state_list = [(self.target_pos[2]-self.sim.pose[2]), self.sim.pose[2], self.sim.v[2]] * self.action_repeat
        state = np.array(state_list) 
        return state