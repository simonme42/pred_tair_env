from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

from gym import error, spaces, utils
from gym.utils import seeding
import gym
import random
import numpy as np

class HVACTairEnv_1846(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self):
        
        super(HVACTairEnv_1846, self).__init__()

        test_data = np.load("env_data/test_data.npy")
        all_data = np.load("env_data/all_data.npy")
        self.start_pred_index = 1844
        self.y_scaler_Tair = joblib.load("env_data/y_scaler_Tair.save") 
        self.y_scaler_Energy = joblib.load("env_data/y_scaler_Energy.save") 
        
        self.model_Tair = Sequential()
        self.model_Tair.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(3, 6)))
        self.model_Tair.add(Dense(1))
        self.model_Tair.compile(optimizer="adam", loss='mse', metrics=['mse'])
        self.model_Tair.load_weights("env_data/LSTM-256-Tair.keras")
        
        self.model_Energy = Sequential()
        self.model_Energy.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(3, 6)))
        self.model_Energy.add(Dense(1))
        self.model_Energy.compile(optimizer="adam", loss='mse', metrics=['mse'])
        self.model_Energy.load_weights("env_data/LSTM-256-Energy_input.keras")
        
        self.min_reward=-25
        self.max_reward = 0
        self.n_actions = 22
        self.max_steps = 168
        self.min_Tvl = 20
        self.max_Tvl = 40
        self.pred_Tair_idx = 0
        self.Setpoint_Temp_idx = 1
        self.Tamb_idx = 2
        self.Rad_hor_idx = 3
        self.Heating_IO_idx = 4
        self.Belegung_idx = 5
        self.Hourofday_idx = 7
        self.Solltemp = 21
        self.Energy_divident = 500
          
        self.data = np.copy(test_data)
        self.all_data = np.copy(all_data)
        self.state = np.copy(self.data[0:1])
        self.reward_range = (self.min_reward, self.max_reward) 
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                            shape=(1,5), 
                                            dtype=np.float32)
        
        ##create the possible actions the agent can take inside the environment
        possible_actions = list((np.arange(self.min_Tvl-1,self.max_Tvl+1)-self.min_Tvl)/(self.max_Tvl-self.min_Tvl))
        for i in range(self.n_actions):
            if i == 0:
                possible_actions[i] = [0.0, 0.0]
            else:
                possible_actions[i] = [possible_actions[i], 1]
        self.possible_actions = possible_actions

    def reset(self):        
        self.current_step = 0 
        # Reset the state of the environment to the initial state
        self.state = np.copy(self.data[self.current_step:self.current_step+1])
        self.current_simtime = self.start_pred_index + 3 + self.current_step
        self.current_daytime = np.copy(self.all_data[self.current_simtime][self.Hourofday_idx])
        self.future_Tamb = np.copy(self.data[self.current_step:self.current_step+1, 2, self.Tamb_idx])
        self.future_Rad = np.copy(self.data[self.current_step:self.current_step+1, 2, self.Rad_hor_idx])
        
        self.tair_at_reset = np.copy(self.state[0, 2, self.pred_Tair_idx])
        self.occupation = np.copy(self.state[0, 2, self.Belegung_idx])
              
        obs = np.zeros((1, 5))

        for i in range(1):
            obs[i, 0] = self.tair_at_reset
            obs[i, 1] = self.occupation
            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1
            obs[i, 3] = np.copy(self.future_Tamb[0])
            obs[i, 4] = np.copy(self.future_Rad[0])        
        
        return obs
                                
    def step(self, action):
        # Execute one time step within the environment
        
        control_params = self.possible_actions[action]
        Setpoint_temp = control_params[0]
        Heating_IO = control_params[1]
        
        self.previous_state = np.copy(self.state[:,:,:])
        self.previous_state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
        self.previous_state[0, 2, self.Heating_IO_idx] = Heating_IO
        
        energy_input_unscaled = self.model_Energy.predict(self.previous_state)
        energy_input = self.y_scaler_Energy.inverse_transform(energy_input_unscaled)
        self.energy_input_reward = energy_input/self.Energy_divident

        self._take_action(action)
        self.current_step += 1
        self.current_simtime += 1
        self.current_daytime = self.all_data[self.current_simtime][self.Hourofday_idx]
        
        self.future_Tamb = self.data[self.current_step:self.current_step+1, 2, self.Tamb_idx]
        self.future_Rad = self.data[self.current_step:self.current_step+1, 2, self.Rad_hor_idx]
        
        tair_after_step_scaled = np.copy(self.state[0, 2, self.pred_Tair_idx])
        tair_after_step = self.y_scaler_Tair.inverse_transform(self.state[0, 2, self.pred_Tair_idx].reshape(-1, 1))[0,0]
        self.Temp_diff = abs(self.Solltemp - tair_after_step)
        self.occupation = self.state[0, 2, self.Belegung_idx]
            
        reward = self._get_reward()
 
        obs = np.zeros((1, 5))
        
        for i in range(1):
            obs[i, 0] = tair_after_step_scaled
            obs[i, 1] = self.occupation
            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1
            obs[i, 3] = np.copy(self.future_Tamb[0])
            obs[i, 4] = np.copy(self.future_Rad[0])
            
        done = self.current_step > self.max_steps
            
        return obs, reward, done, {}
    
    def _get_reward(self):
        
        if self.occupation==1:
            reward = - (self.Temp_diff * 10)
        else:
            reward = np.min(-self.energy_input_reward,0)
        
        return reward
    
    def _take_action(self, action):
        control_action = self.possible_actions[action]
        Setpoint_temp = control_action[0]
        Heating_IO = control_action[1]
                
        if self.current_step == 0:        
            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO
            tair_next_step = self.model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step
            
            self.state = next_state
            
        elif self.current_step == 1:
            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO
            tair_next_step = self.model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, self.Setpoint_Temp_idx] = self.state[0, 1, self.Setpoint_Temp_idx]
            next_state[0, 0, self.Heating_IO_idx] = self.state[0, 1, self.Heating_IO_idx]
            next_state[0, 1, self.pred_Tair_idx] = self.state[0, 2, self.pred_Tair_idx]           
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step
            
            self.state = next_state

        else:
            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO 
            tair_next_step = self.model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, self.Setpoint_Temp_idx] = self.state[0, 1, self.Setpoint_Temp_idx]
            next_state[0, 0, self.Heating_IO_idx] = self.state[0, 1, self.Heating_IO_idx]
            next_state[0, 0, self.pred_Tair_idx] = self.state[0, 1, self.pred_Tair_idx]           
            next_state[0, 1, self.pred_Tair_idx] = self.state[0, 2, self.pred_Tair_idx]
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO            
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step
            
            self.state = next_state
            

            
  class HVACTairEnv_7390(gym.Env):

    metadata = {'render.modes': ['human']}

  

    def __init__(self):

        super(HVACTairEnv_7390, self).__init__()

        test_data = np.load("env_data_V2/test_data.npy")

        all_data = np.load("env_data_V2/all_data.npy")

        self.start_pred_index = 7388

        self.y_scaler_Tair = joblib.load("env_data_V2/y_scaler_Tair.save") 

        self.y_scaler_Energy = joblib.load("env_data_V2/y_scaler_Energy.save") 

        

        self.model_Tair = Sequential()

        self.model_Tair.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(3, 6)))

        self.model_Tair.add(Dense(1))

        self.model_Tair.compile(optimizer="adam", loss='mse', metrics=['mse'])

        self.model_Tair.load_weights("env_data_V2/LSTM-256-Tair.keras")

        

        self.model_Energy = Sequential()

        self.model_Energy.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(3, 6)))

        self.model_Energy.add(Dense(1))

        self.model_Energy.compile(optimizer="adam", loss='mse', metrics=['mse'])

        self.model_Energy.load_weights("env_data_V2/LSTM-256-Energy_input.keras")

        

        self.min_reward=-25

        self.max_reward = 0

        self.n_actions = 22

        self.max_steps = 168

        self.min_Tvl = 20

        self.max_Tvl = 40

        self.pred_Tair_idx = 0

        self.Setpoint_Temp_idx = 1

        self.Tamb_idx = 2

        self.Rad_hor_idx = 3

        self.Heating_IO_idx = 4

        self.Belegung_idx = 5

        self.Hourofday_idx = 7

        self.Solltemp = 21

        self.Energy_divident = 500

          

        self.data = np.copy(test_data)

        self.all_data = np.copy(all_data)

        self.state = np.copy(self.data[0:1])

        self.reward_range = (self.min_reward, self.max_reward) 

        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Box(low=0.0, high=1.0, 

                                            shape=(1,5), 

                                            dtype=np.float32)

        

        ##create the possible actions the agent can take inside the environment

        possible_actions = list((np.arange(self.min_Tvl-1,self.max_Tvl+1)-self.min_Tvl)/(self.max_Tvl-self.min_Tvl))

        for i in range(self.n_actions):

            if i == 0:

                possible_actions[i] = [0.0, 0.0]

            else:

                possible_actions[i] = [possible_actions[i], 1]

        self.possible_actions = possible_actions



    def reset(self):        

        self.current_step = 0 

        # Reset the state of the environment to the initial state

        self.state = np.copy(self.data[self.current_step:self.current_step+1])

        self.current_simtime = self.start_pred_index + 3 + self.current_step

        self.current_daytime = np.copy(self.all_data[self.current_simtime][self.Hourofday_idx])

        self.future_Tamb = np.copy(self.data[self.current_step:self.current_step+1, 2, self.Tamb_idx])

        self.future_Rad = np.copy(self.data[self.current_step:self.current_step+1, 2, self.Rad_hor_idx])

        

        self.tair_at_reset = np.copy(self.state[0, 2, self.pred_Tair_idx])

        self.occupation = np.copy(self.state[0, 2, self.Belegung_idx])

              

        obs = np.zeros((1, 5))



        for i in range(1):

            obs[i, 0] = self.tair_at_reset

            obs[i, 1] = self.occupation

            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1

            obs[i, 3] = np.copy(self.future_Tamb[0])

            obs[i, 4] = np.copy(self.future_Rad[0])        

        

        return obs

                                

    def step(self, action):

        # Execute one time step within the environment

        

        control_params = self.possible_actions[action]

        Setpoint_temp = control_params[0]

        Heating_IO = control_params[1]

        

        self.previous_state = np.copy(self.state[:,:,:])

        self.previous_state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp

        self.previous_state[0, 2, self.Heating_IO_idx] = Heating_IO

        

        energy_input_unscaled = self.model_Energy.predict(self.previous_state)

        energy_input = self.y_scaler_Energy.inverse_transform(energy_input_unscaled)

        self.energy_input_reward = energy_input/self.Energy_divident



        self._take_action(action)

        self.current_step += 1

        self.current_simtime += 1

        self.current_daytime = self.all_data[self.current_simtime][self.Hourofday_idx]

        

        self.future_Tamb = self.data[self.current_step:self.current_step+1, 2, self.Tamb_idx]

        self.future_Rad = self.data[self.current_step:self.current_step+1, 2, self.Rad_hor_idx]

        

        tair_after_step_scaled = np.copy(self.state[0, 2, self.pred_Tair_idx])

        tair_after_step = self.y_scaler_Tair.inverse_transform(self.state[0, 2, self.pred_Tair_idx].reshape(-1, 1))[0,0]

        self.Temp_diff = abs(self.Solltemp - tair_after_step)

        self.occupation = self.state[0, 2, self.Belegung_idx]

            

        reward = self._get_reward()

 

        obs = np.zeros((1, 5))

        

        for i in range(1):

            obs[i, 0] = tair_after_step_scaled

            obs[i, 1] = self.occupation

            obs[i, 2] = (self.current_daytime-1) / 23  #scale between 0 and 1

            obs[i, 3] = np.copy(self.future_Tamb[0])

            obs[i, 4] = np.copy(self.future_Rad[0])

            

        done = self.current_step > self.max_steps
           
        return obs, reward, done, {}   

    def _get_reward(self):
        
        if self.occupation==1:

            reward = - (self.Temp_diff * 10)

        else:

            reward = np.min(-self.energy_input_reward,0)        

        return reward    

    def _take_action(self, action):

        control_action = self.possible_actions[action]
        Setpoint_temp = control_action[0]
        Heating_IO = control_action[1]                

        if self.current_step == 0:        

            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO

            tair_next_step = self.model_Tair.predict(self.state)

            

            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step

            

            self.state = next_state

            

        elif self.current_step == 1:

            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO

            tair_next_step = self.model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, self.Setpoint_Temp_idx] = self.state[0, 1, self.Setpoint_Temp_idx]
            next_state[0, 0, self.Heating_IO_idx] = self.state[0, 1, self.Heating_IO_idx]
            next_state[0, 1, self.pred_Tair_idx] = self.state[0, 2, self.pred_Tair_idx]           
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step

            

            self.state = next_state



        else:

            self.state[0, 2, self.Setpoint_Temp_idx] = Setpoint_temp
            self.state[0, 2, self.Heating_IO_idx] = Heating_IO 

            tair_next_step = self.model_Tair.predict(self.state)
            
            next_state = self.data[self.current_step+1:self.current_step+2]
            next_state[0, 0, self.Setpoint_Temp_idx] = self.state[0, 1, self.Setpoint_Temp_idx]
            next_state[0, 0, self.Heating_IO_idx] = self.state[0, 1, self.Heating_IO_idx]
            next_state[0, 0, self.pred_Tair_idx] = self.state[0, 1, self.pred_Tair_idx]           
            next_state[0, 1, self.pred_Tair_idx] = self.state[0, 2, self.pred_Tair_idx]
            next_state[0, 1, self.Setpoint_Temp_idx] = Setpoint_temp
            next_state[0, 1, self.Heating_IO_idx] = Heating_IO            
            next_state[0, 2, self.pred_Tair_idx] = tair_next_step
            
            self.state = next_state
