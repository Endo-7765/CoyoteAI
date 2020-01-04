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
from estimation_model import EstimationModelByContinuous
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(42,200)
    nn.init.kaiming_normal_(self.fc1.weight)
    self.fc2 = nn.Linear(200,200)
    nn.init.kaiming_normal_(self.fc2.weight)
    self.fc3 = nn.Linear(200,200)
    nn.init.kaiming_normal_(self.fc3.weight)
    self.fc4 = nn.Linear(200,200)
    nn.init.kaiming_normal_(self.fc4.weight)
    self.fc5 = nn.Linear(200,200)
    nn.init.kaiming_normal_(self.fc5.weight)
    self.fc6 = nn.Linear(200,200)
    nn.init.kaiming_normal_(self.fc6.weight)
    self.fc7 = nn.Linear(200,62)
    nn.init.kaiming_normal_(self.fc7.weight)

  def forward(self,x):
    h = F.relu(self.fc1(x))
    h = F.relu(self.fc2(h))
    h = F.relu(self.fc3(h))
    h = F.relu(self.fc4(h))
    h = F.relu(self.fc5(h))
    h = F.relu(self.fc6(h))
    h = self.fc7(h)
    return h
class DQNPlayerConfiguration:#DQNPlayerの学習のパラメータを管理
  def __init__(self):
    self.MEMORY_SIZE = 500
    self.BATCH_SIZE = 50
    self.EPSILON = 1.0
    self.EPSILON_DECREASE = 1e-5 # εの減少値
    self.EPSILON_MIN = 0.1 # εの下限
    self.START_REDUCE_EPSILON = 200 # εを減少させるステップ数
    self.TRAIN_FREQ = 10 # Q関数の学習間隔
    self.GAMMA = 0.4
    self.UPDATE_TARGET_Q_FREQ = 20
    self.START_Q_LEARNING = 30000
class DQNPlayer(Player):
  def __init__(self,configuration = DQNPlayerConfiguration(),loss_output = None):
    super(DQNPlayer,self).__init__()
    self.configuration = configuration
    if loss_output is not None:
      self.loss_output = open(loss_output,mode='w')
    else:
      self.loss_output = None
    self.Q = Net()#Q_value estimate network
    self.Q_ast = copy.deepcopy(self.Q)#target network
    self.card_estimator  = EstimationModelByContinuous()
    self.optimizer = optim.SGD(self.Q.parameters(),lr = 1e-1,momentum=0.3)
    self.card_estimator_optimizer = optim.SGD(self.card_estimator.parameters(),lr=1e-1,momentum=0.3)
    self.memory=[]#リプレイ用の配列(state,action,reward,next_state,done)
    self.card_estimator_memory = []#カード推定器の学習用のリプレイ配列(history,answer)
    self.EPSILON = self.configuration.EPSILON
    self.temp_memory = []#直前の行動(state,action)を覚える。次に__call__が呼び出されたときに、memoryに追加する
    self.card_estimator_temp_memory = []#前回の答え合わせ以降の状態を全て保存
    self.sum_reward = 0.0#前回の__call__以降に受け取った報酬の和
    self.done=False
    self.my_card = None
    self.count = 0#何回__call__が呼び出されたか
    self.card_estimator_temp_count = 0
    self.evaluation_mode = False#評価用モードでは、常に最適な手を取り続ける
  def __call__(self,history):
    #Add something to memory
    state_vector = self.createInputToNet(history).reshape(1,28)#numpy (1,28)
    #まず自分のカードの推定器を用いる。
    my_card_probability = self.card_estimator(torch.from_numpy(state_vector))#tensor dim(1,14)
    input_to_net = np.concatenate([F.softmax(my_card_probability,dim=1).detach().numpy(),state_vector],axis=1)#numpy dim(1,42)
    if len(self.card_estimator_temp_memory) != 0 and self.done:#前回の試行から、一回のプレイが終わり、正解が明らかになった。;
      for j in self.card_estimator_temp_memory:
        self.card_estimator_memory.append((j,self.my_card.to_onehot()))
        if len(self.card_estimator_memory) > self.configuration.MEMORY_SIZE:
          self.card_estimator_memory.pop(0)
      self.card_estimator_temp_memory = []
      self.my_card = None
    if len(self.card_estimator_memory)>=self.configuration.MEMORY_SIZE and (self.count+int(self.configuration.TRAIN_FREQ/2))%self.configuration.TRAIN_FREQ == 0:
      self.card_estimator_train()
      

    if len(self.temp_memory)!=0:
      self.memory.append((self.temp_memory[0],self.temp_memory[1],self.sum_reward,input_to_net,self.done))
      if len(self.memory)>self.configuration.MEMORY_SIZE:
        self.memory.pop(0)#過去の記録を削除
        if self.count % self.configuration.TRAIN_FREQ == 0 and self.count > self.configuration.START_Q_LEARNING:
          self.train()
    #EPSILONを変える
    if self.count > self.configuration.START_REDUCE_EPSILON:
      self.EPSILON = max(self.EPSILON - self.configuration.EPSILON_DECREASE,self.configuration.EPSILON_MIN)
    retval = 0#save the retval in this function
    if self.evaluation_mode or np.random.rand()>self.EPSILON:
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
    self.card_estimator_temp_memory.append(state_vector)
    self.count += 1
    return retval
          
      

      
  def createInputToNet(self,history):
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
    for i,history_element in enumerate(reversed(history)):
      if i>=4:
        break
      retval_history[i]=history_element*NORMALIZE_MULTIPLIER
      retval_history[i+4]=1.0
    return np.concatenate([retval_cards,retval_history])
  def learn(self,result,my_card,history):
    self.sum_reward += result
    if self.my_card == None:
      self.my_card = my_card
    self.done=True
  def train(self):
    if self.evaluation_mode:
      return
    memory_ = np.random.permutation(self.memory)
    memory_idx = range(len(memory_))
    for i in memory_idx[::self.configuration.BATCH_SIZE]:
      batch = np.array(memory_[i:i+self.configuration.BATCH_SIZE]) # 経験ミニバッチ
      states = np.array(batch[:,0].tolist(), dtype="float32").reshape((self.configuration.BATCH_SIZE,42))
      acts = np.array(batch[:,1].tolist(), dtype="int32")
      rewards = np.array(batch[:,2].tolist(), dtype="float32")
      next_states = np.array(batch[:,3].tolist(), dtype="float32").reshape((self.configuration.BATCH_SIZE, 42))
      dones = np.array(batch[:,4].tolist(), dtype="bool")
      
      q = self.Q(torch.from_numpy(states))
      with torch.no_grad():
        maxs = np.max(self.Q_ast(torch.from_numpy(next_states)).numpy(),axis=1)
      target = copy.deepcopy(q.data.numpy())
      for j in range(self.configuration.BATCH_SIZE):
        target[j,acts[j]]=rewards[j]+self.configuration.GAMMA * maxs[j]*(not dones[j])
      self.optimizer.zero_grad()
      loss = nn.SmoothL1Loss()(q,torch.from_numpy(target))

      loss.backward()
      self.optimizer.step()
    if self.count % self.configuration.UPDATE_TARGET_Q_FREQ==0:
      self.Q_ast = copy.deepcopy(self.Q)
  def card_estimator_train(self):
    average_loss = 0.0
    memory_ = np.random.permutation(self.card_estimator_memory)
    memory_idx = range(len(memory_))
    self.card_estimator_temp_count += 1
    for i in memory_idx[::self.configuration.BATCH_SIZE]:
      batch = np.array(memory_[i:i+self.configuration.BATCH_SIZE])
      states = np.array(batch[:,0].tolist(),dtype='float32').reshape(self.configuration.BATCH_SIZE,28)
      answer = np.array(batch[:,1].tolist(),dtype='int64')

      self.card_estimator_optimizer.zero_grad()
      estimation = self.card_estimator(torch.from_numpy(states))
      loss = nn.CrossEntropyLoss()(estimation,torch.from_numpy(answer))
      average_loss += loss.detach().numpy()
      if not self.evaluation_mode:
        loss.backward()
        self.card_estimator_optimizer.step()
    average_loss /= (len(memory_)/self.configuration.BATCH_SIZE)
    if self.loss_output is not None:
      self.loss_output.write(str(self.count)+","+str(average_loss)+'\n')
   
  def __del__(self):
    if self.loss_output is None:
      self.loss_output.close()
    
