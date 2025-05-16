import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import gc
from torch.cuda.amp import autocast, GradScaler

# FOR GPU ONLINE
GPU_MEMORY_FRACTION = float(os.environ.get('GPU_MEMORY_FRACTION', '1'))  
USE_MIXED_PRECISION = os.environ.get('USE_MIXED_PRECISION', 'True').lower() == 'true'
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', '1'))
DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', '64'))

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fci1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fci1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        #  batch normalization 
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        
        # optimizer +  weight decay 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.loss = nn.MSELoss()
        
        #memory managment
        self.device = T.device('cuda:0')
        
    
        self.to(self.device)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight) 
        
        
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)  
            single_sample = True
        else:
            single_sample = False
            
        x = self.fc1(state)
        
        if self.training or not single_sample:
            x = self.bn1(x)
            
        x = F.relu(x)
        
        x = self.fc2(x)
        if self.training or not single_sample:
            x = self.bn2(x)
            
        x = F.relu(x)
        
        actions = self.fc3(x)
        
        return actions
    
class Agent():
    #HYPERPARAMETERS
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size=None, n_actions=None, 
                max_mem_size=1000000, eps_end=0.01, eps_dec=0.000007, 
                load_model=False, target_update=100, model_dir='models',
                use_mixed_precision=USE_MIXED_PRECISION, grad_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
        
        self.gamma = gamma
        self.epsilon = epsilon  
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        self.n_actions = n_actions if n_actions is not None else 5  
        self.action_space = [i for i in range(self.n_actions)]  
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0  
        self.target_update = target_update
        self.learn_step_counter = 0
        self.model_dir = model_dir
        
        self.use_mixed_precision = use_mixed_precision
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_step_counter = 0
        
        #CUDA GPU AVAILABLE
        self.scaler = GradScaler() if self.use_mixed_precision and T.cuda.is_available() else None
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.Q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=input_dims,
                                  fci1_dims=256, fc2_dims=256)
        
        self.Q_target = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=input_dims,
                                    fci1_dims=256, fc2_dims=256)
        
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
        
        self.Q_target.eval()
        
        # detect device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            # print GPU info
            print(f"GPU: {T.cuda.get_device_name(0)}")
            print(f"Memory allocated: {T.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory reserved: {T.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            #memory fraction
            if hasattr(T.cuda, 'set_per_process_memory_fraction'):
                T.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
                print(f"Set GPU memory fraction to {GPU_MEMORY_FRACTION}")
        
        if load_model:
            self.load_model()

       
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float32).to(self.device)
            
            self.Q_eval.eval()
            
            with T.no_grad():
                if self.use_mixed_precision and self.device.type == 'cuda':
                    with autocast():
                        actions = self.Q_eval.forward(state)
                else:
                    actions = self.Q_eval.forward(state)
            
            self.Q_eval.train()
            
            action = T.argmax(actions).item()
        else: 
            action = np.random.choice(self.action_space)

        return action 
    
    def _calculate_loss(self, state_batch, action_batch, reward_batch, new_state_batch, terminal_batch, batch_index):
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        with T.no_grad():
            q_next = self.Q_target.forward(new_state_batch)
        
        
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        
        return loss
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
       
        do_update = (self.grad_step_counter % self.grad_accumulation_steps == 0)
        
        if do_update:
            self.Q_eval.optimizer.zero_grad()
        
      
        self.learn_step_counter += 1
        self.grad_step_counter += 1
        
       
        if self.learn_step_counter % self.target_update == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            print(f"Target network updated at step {self.learn_step_counter}")
    
            if self.device.type == 'cuda':
                T.cuda.empty_cache()

      
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        

        state_batch = T.tensor(self.state_memory[batch], dtype=T.float32).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(self.device)
        action_batch = self.action_memory[batch]


        if self.use_mixed_precision and self.device.type == 'cuda':
            with autocast():
                loss = self._calculate_loss(
                    state_batch, action_batch, reward_batch, new_state_batch, terminal_batch, batch_index
                )
                loss = loss / self.grad_accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if do_update:
                self.scaler.step(self.Q_eval.optimizer)
                self.scaler.update()
        else:
            loss = self._calculate_loss(
                state_batch, action_batch, reward_batch, new_state_batch, terminal_batch, batch_index
            )

            loss = loss / self.grad_accumulation_steps
            
            loss.backward()
            
            if do_update:
                self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
        
        if self.learn_step_counter % 1000 == 0 and self.device.type == 'cuda':
            T.cuda.empty_cache()
            gc.collect()  

    def save_model(self, filename=None):
        """Save model with all training state"""
        if filename is None:
            filename = os.path.join(self.model_dir, 'agent_model.pth')
        
        self.Q_eval.cpu()
        self.Q_target.cpu()
        
        save_data = {
            'Q_eval_state_dict': self.Q_eval.state_dict(),
            'Q_target_state_dict': self.Q_target.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'grad_step_counter': self.grad_step_counter,
            'optimizer_state': self.Q_eval.optimizer.state_dict(),
            'input_dims': self.input_dims,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'target_update': self.target_update
        }
        
        print(f"Saving model to {filename}")
        T.save(save_data, filename)
        
        self.Q_eval.to(self.device)
        self.Q_target.to(self.device)
    
    def load_model(self, filename=None):
        """Load model with improved error handling and compatibility"""
        if filename is None:
            filename = os.path.join(self.model_dir, 'agent_model.pth')
            
        try:
            if os.path.isfile(filename):
                checkpoint = T.load(filename, map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'Q_eval_state_dict' in checkpoint:
                    self.Q_eval.load_state_dict(checkpoint['Q_eval_state_dict'])
                    self.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
                    self.epsilon = checkpoint.get('epsilon', self.epsilon)
                    self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
                    self.grad_step_counter = checkpoint.get('grad_step_counter', 0)
                    
                    if ('optimizer_state' in checkpoint and 
                        checkpoint.get('n_actions') == self.n_actions and
                        checkpoint.get('input_dims') == self.input_dims):
                        self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    
                    if 'gamma' in checkpoint:
                        self.gamma = checkpoint['gamma']
                    if 'target_update' in checkpoint:
                        self.target_update = checkpoint['target_update']
                    
                    print(f"Loaded model from {filename} (new format)")
                else:
                    self.Q_eval.load_state_dict(checkpoint)
                    self.Q_target.load_state_dict(checkpoint)
                    print(f"Loaded model from {filename} (old format)")
                
                self.Q_eval.to(self.device)
                self.Q_target.to(self.device)
                
                self.Q_target.eval()
            else: 
                print(f"No model file found: {filename}")
        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")
            self.Q_eval.to(self.device)
            self.Q_target.to(self.device)