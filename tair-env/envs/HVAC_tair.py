import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
from gym import spaces
import numpy as np

class pred_Tair_env(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self, data, all_data, start_pred_index, min_reward, max_reward, n_actions, 
                max_steps=168, Hourofday_idx, Tamb_idx, Rad_hor_idx, 
                pred_Tair_idx, Belegung_idx, Setpoint_Temp_idx,
                Air_flow_rate_idx, y_scaler_Tair, y_scaler_Energy, 
                model_Tair, model_Energy_input, Solltemp=21, Energy_divident=500):
        
        super(pred_Tair_env, self).__init__()
        
        self.data = np.copy(data)
        self.all_data = all_data
        self.state = np.copy(self.data[0:1])
        self.reward_range = (min_reward, max_reward) 
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                            shape=(1,5), 
                                            dtype=np.float32)
        ##create the possible actions the agent can take inside the environment
        possible_actions = list((np.arange(19,41)-20)/20)

        for i in range(n_actions):
            if i == 0:
                possible_actions[i] = [0.0, 0.0]
            else:
                possible_actions[i] = [possible_actions[i], 1]

        self.possible_actions = possible_actions[:]

    def reset(self):        
        self.current_step = 0 
        # Reset the state of the environment to the initial state
        self.state = np.copy(self.data[self.current_step:self.current_step+1])
        self.current_simtime = start_pred_index + 3 + self.current_step
        self.current_daytime = all_data[self.current_simtime][Hourofday_idx]
        self.future_Tamb = self.data[self.current_step:self.current_step+6, 2, Tamb_idx]
        self.future_Rad = self.data[self.current_step:self.current_step+6, 2, Rad_hor_idx]
        tair_after_step_scaled = np.copy(self.state[0, 2, pred_Tair_idx])
        
        occupation = self.state[0, 2, Belegung_idx]
        
        raw_obs = self.state[0,-1:,:]        
        obs = np.zeros((1, 5))
        
        #if self.current_daytime > occupy_start and self.current_daytime <= occupy_end:
            #occupation += 1

        for i in range(1):
            obs[i, 0] = tair_after_step_scaled
            obs[i, 1] = occupation
            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1
            obs[i, 3] = np.copy(self.future_Tamb[0])
            obs[i, 4] = np.copy(self.future_Rad[0])
        
        
        return obs
                                
    def step(self, action):
        # Execute one time step within the environment
        control_params = possible_actions[action]
        heating_IO = control_params[1]
        self.previous_state = np.copy(self.state[:,:,:])
        self.previous_state[0, 2, Setpoint_Temp_idx] = control_params[0]
        self.previous_state[0, 2, Air_flow_rate_idx] = control_params[1]
        energy_input_unscaled = model_Energy_input.predict(self.previous_state)
        energy_input = y_scaler_Energy.inverse_transform(energy_input_unscaled)
        self.energy_input_reward = energy_input/Energy_divident

        self._take_action(action)
        self.current_step += 1
        self.current_simtime += 1
        self.current_daytime = all_data[self.current_simtime][Hourofday_idx]
        
        self.future_Tamb = self.data[self.current_step:self.current_step+1, 2, Tamb_idx]
        self.future_Rad = self.data[self.current_step:self.current_step+1, 2, Rad_hor_idx]
        
        tair_after_step_scaled = np.copy(self.state[0, 2, pred_Tair_idx])
        tair_after_step = y_scaler_Tair.inverse_transform(self.state[0, 2, pred_Tair_idx].reshape(-1, 1))[0,0]
        self.Temp_diff = abs(Solltemp - tair_after_step)
        self.occupation = self.state[0, 2, Belegung_idx]
            
        reward = self._get_reward(self)
 
        obs = np.zeros((1, 5))
        
        for i in range(n_steps_obs):
            obs[i, 0] = tair_after_step_scaled
            obs[i, 1] = occupation
            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1
            obs[i, 3] = np.copy(self.future_Tamb[0])
            obs[i, 4] = np.copy(self.future_Rad[0])
            
        done = self.current_step > max_steps
            
        return obs, reward, done, {}
    
    def _get_reward(self):
        
        if self.occupation==1:
            reward = - (self.Temp_diff * 10)
        else:
            reward = np.min(-self.energy_input_reward,0)
        
        return reward
    
    def _take_action(self, action):
        control_action = possible_actions[action]
        setpoint_temp = control_action[0]
        air_flow_IO = control_action[1]
                
        if self.current_step == 0:        
            self.state[0, 2, Setpoint_Temp_idx] = setpoint_temp
            self.state[0, 2, Air_flow_rate_idx] = air_flow_IO
            tair_next_step = model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 1, Setpoint_Temp_idx] = setpoint_temp
            next_state[0, 1, Air_flow_rate_idx] = air_flow_IO
            next_state[0, 2, pred_Tair_idx] = tair_next_step
            
            self.state = next_state
            
        elif self.current_step == 1:
            self.state[0, 2, Setpoint_Temp_idx] = setpoint_temp
            self.state[0, 2, Air_flow_rate_idx] = air_flow_IO

            tair_next_step = model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, Setpoint_Temp_idx] = self.state[0, 1, Setpoint_Temp_idx]
            next_state[0, 0, Air_flow_rate_idx] = self.state[0, 1, Air_flow_rate_idx]
            next_state[0, 1, pred_Tair_idx] = self.state[0, 2, pred_Tair_idx]           
            next_state[0, 1, Setpoint_Temp_idx] = setpoint_temp
            next_state[0, 1, Air_flow_rate_idx] = air_flow_IO
            next_state[0, 2, pred_Tair_idx] = tair_next_step
            
            self.state = next_state

        else:
            self.state[0, 2, Setpoint_Temp_idx] = setpoint_temp
            self.state[0, 2, Air_flow_rate_idx] = air_flow_IO
 
            tair_next_step = model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, Setpoint_Temp_idx] = self.state[0, 1, Setpoint_Temp_idx]
            next_state[0, 0, Air_flow_rate_idx] = self.state[0, 1, Air_flow_rate_idx]
            next_state[0, 0, pred_Tair_idx] = self.state[0, 1, pred_Tair_idx]           
            next_state[0, 1, pred_Tair_idx] = self.state[0, 2, pred_Tair_idx]
            next_state[0, 1, Setpoint_Temp_idx] = setpoint_temp
            next_state[0, 1, Air_flow_rate_idx] = air_flow_IO            
            next_state[0, 2, pred_Tair_idx] = tair_next_step
            
            self.state = next_state
            
