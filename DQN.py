from player import *
from main import *
from card import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(42,200)
    self.fc2 = nn.Linear(200,200)
    self.fc3 = nn.Linear(200,200)
    self.fc4 = nn.Linear(200,200)
    self.fc5 = nn.Linear(200,62)
  def forward(self,x):
    h = F.relu(self.fc1(x))
    h = F.relu(self.fc2(h))
    h = F.relu(self.fc3(h))
    h = F.relu(self.fc4(h))
    h = self.fc5(h)
    return h
MEMORY_SIZE = 500
BATCH_SIZE = 50
EPSILON = 1.0
EPSILON_DECREASE = 0.0001 # εの減少値
EPSILON_MIN = 0.1 # εの下限
START_REDUCE_EPSILON = 200 # εを減少させるステップ数
TRAIN_FREQ = 10 # Q関数の学習間隔
GAMMA = 0.4
UPDATE_TARGET_Q_FREQ = 20
class DQNPlayer(Player):
  def __init__(self):
    super(DQNPlayer,self).__init__()
    self.Q = Net()
    self.Q_ast = copy.deepcopy(self.Q)
    self.optimizer = optim.SGD(self.Q.parameters(),lr = 1e-3,momentum=0.9)
    self.memory=[]#リプレイ用の配列(state,action,reward,next_state,done)
    self.EPSILON = EPSILON
    self.temp_memory = []#直前の行動(state,action)を覚える。次に__call__が呼び出されたときに、memoryに追加する
    self.sum_reward = 0.0#前回の__call__以降に受け取った報酬の和
    self.done=False
    self.count = 0#何回__call__が呼び出されたか
  def __call__(self,history):
    #Add something to memory
    input_to_net = self.createInputToNet().reshape(1,42)
    if len(self.temp_memory)!=0:
      self.memory.append((self.temp_memory[0],self.temp_memory[1],self.sum_reward,input_to_net,self.done))
      if len(self.memory)>MEMORY_SIZE:
        self.memory.pop(0)#過去の記録を削除
        if self.count % TRAIN_FREQ == 0:
          self.train()
    self.count += 1
    #EPSILONを変える
    if self.count > START_REDUCE_EPSILON:
      self.EPSILON = max(self.EPSILON - EPSILON_DECREASE,EPSILON_MIN)
        
    retval = 0#save the retval in this function
    if np.random.rand()>self.EPSILON:
      #最適な行動を行う
      with torch.no_grad():
        Q_value_action = self.Q(torch.from_numpy(input_to_net)).numpy().ravel()#shape:(62)
      if len(history)!=0 and history[-1]>=60:#現在のネットワークでは、最大60までの値しか答えないため、それ以上のときは必ずコヨーテ
        retval = -100
      else:
        if len(history)==0:
          max_value_action = np.argmax(Q_value_action)
        else:
          max_value_action = history[-1]+1+np.argmax(Q_value_action[history[-1]+1:])
        if max_value_action == 61:
          retval = -100
        else:
          retval = max_value_action
    else:
      #ランダムに行動を選択
      if np.random.randint(0,2)==0:
        retval = -100
      else:
        if len(history)==0:
          retval = np.random.randint(0,61)
        elif history[-1]>=60:
          retval = -100
        else:
          retval = np.random.randint(history[-1]+1,61)
    #今回取った行動を記録しておく
    if retval == -100:
      self.temp_memory = (input_to_net,61)
    else:
      self.temp_memory = (input_to_net,retval)
    self.sum_reward = 0.0
    self.done=False
    return retval
          
      

      
  def createInputToNet(self):
    NORMALIZE_MULTIPLIER = 0.05
    retval_cards = []
    for i in self.cards:
      if i.flag == 0:
        retval_cards.append(np.array([i.num*NORMALIZE_MULTIPLIER,1,0,0,0],dtype=np.float32))
      else:
        temp = np.zeros(5,dtype=np.float32)
        temp[i.flag+1]=1.0
        retval_cards.append(temp)
    retval_cards = np.concatenate(retval_cards)
    retval_history=np.zeros(8,dtype=np.float32)
    for i,history_element in enumerate(self.history):
      retval_history[i]=history_element*NORMALIZE_MULTIPLIER
      retval_history[i+4]=1.0
    retval_mycard = np.ones(14,dtype=np.float32)/14.0
    return np.concatenate([retval_mycard,retval_cards,retval_history])
  def learn(self,result,history):
    self.sum_reward += result
    self.done=True
  def train(self):
    memory_ = np.random.permutation(self.memory)
    memory_idx = range(len(memory_))
    for i in memory_idx[::BATCH_SIZE]:
      batch = np.array(memory_[i:i+BATCH_SIZE]) # 経験ミニバッチ
      states = np.array(batch[:,0].tolist(), dtype="float32").reshape((BATCH_SIZE,42))
      acts = np.array(batch[:,1].tolist(), dtype="int32")
      rewards = np.array(batch[:,2].tolist(), dtype="float32")
      next_states = np.array(batch[:,3].tolist(), dtype="float32").reshape((BATCH_SIZE, 42))
      dones = np.array(batch[:,4].tolist(), dtype="bool")
      
      q = self.Q(torch.from_numpy(states))
      with torch.no_grad():
        maxs = np.max(self.Q_ast(torch.from_numpy(next_states)).numpy(),axis=1)
      target = copy.deepcopy(q.data.numpy())
      for j in range(BATCH_SIZE):
        target[j,acts[j]]=rewards[j]+GAMMA * maxs[j]*(not dones[j])
      self.optimizer.zero_grad()
      loss = nn.SmoothL1Loss()(q,torch.from_numpy(target))

      loss.backward()
      self.optimizer.step()
    if self.count % UPDATE_TARGET_Q_FREQ==0:
      self.Q_ast = copy.deepcopy(self.Q)




      
