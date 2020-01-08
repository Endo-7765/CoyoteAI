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


class BaysianEstimatorPlayerConfiguration:
  def __init__(self):
    self.MEMORY_SIZE = 500
    self.BATCH_SIZE = 50
    self.TRAIN_FREQ = 10
    self.onehot_to_card = [Card(20,0),
      Card(15,0),
      Card(10,0),
      Card(5,0),
      Card(4,0),
      Card(3,0),
      Card(2,0),
      Card(1,0),
      Card(0,0),
      Card(-5,0),
      Card(-10,0),
      Card(0,1),
      Card(0,2),
      Card(0,3)]
class BaysianEstimatorPlayer(Player):
  def __init__(self,defensive_rate,coyote_rate,configuration = BaysianEstimatorPlayerConfiguration(),loss_output = None):
    self.defensive_rate = defensive_rate
    self.coyote_rate = coyote_rate
    if loss_output is not None:
      self.loss_output = open(loss_output,mode='w')
    else:
      self.loss_output = None
    self.configuration = configuration
    self.card_estimator = EstimationModelByContinuous()
    self.card_estimator_optimizer = optim.SGD(self.card_estimator.parameters(),lr=1e-1,momentum = 0.3,weight_decay = 1e-3)
    self.card_estimator_memory = []#カード推定器の学習用のリプレイ配列(history,answer)
    self.card_estimator_temp_memory = []#前回の答え合わせ以降の状態を全て保存
    self.done=False
    self.my_card = None
    self.count = 0
    self.evaluation_mode = False
  def __call__(self,history):
    state_vector = self.createInputToNet(history).reshape(1,28) #numpy (1,28)
    my_card_probability = self.card_estimator(torch.from_numpy(state_vector))#torch tensor (1,14)
    if len(self.card_estimator_temp_memory)!= 0 and self.done: #前回の試行から、一回のプレイが終わり、正解が明らかになった
      for j in self.card_estimator_temp_memory:
        self.card_estimator_memory.append((j,self.my_card.to_onehot()))
        if len(self.card_estimator_memory) > self.configuration.MEMORY_SIZE:
          self.card_estimator_memory.pop(0)
      self.card_estimator_temp_memory = []
      self.my_card = None
    if len(self.card_estimator_memory)>=self.configuration.MEMORY_SIZE and self.count%self.configuration.TRAIN_FREQ == 0:
      self.card_estimator_train()
      self.card_estimator_memory = []
    self.done=False
    self.card_estimator_temp_memory.append(state_vector)
    self.count += 1

    #calculate the true sum by my_card_probability
    my_card_probability = F.softmax(my_card_probability[:,:-1],dim=1).detach().numpy().ravel()#numpy (13,)
    estimated_value = np.array([calcsum(self.cards+[i]) for i in self.configuration.onehot_to_card[:-1]],dtype=np.int64) #?カードはこの計算では無視 numpy(13,)
    estimated_value_sorted,index_sorted = np.sort(estimated_value),np.argsort(estimated_value)

    probability_sorted_cumsum = np.cumsum(my_card_probability[index_sorted])

    defensive_threshold = estimated_value_sorted[min(np.searchsorted(probability_sorted_cumsum,self.defensive_rate),len(probability_sorted_cumsum)-1)]
    coyote_threshold = estimated_value_sorted[min(np.searchsorted(probability_sorted_cumsum,self.coyote_rate),len(probability_sorted_cumsum)-1)]

    if len(history)==0:#自分が最初の着手
      if coyote_threshold <0:
        return -100
      elif defensive_threshold >=0:
        return defensive_threshold
      else:
        return 0
    else:
      if coyote_threshold < history[-1]:
        return -100
      elif history[-1] <  defensive_threshold:
        return defensive_threshold
      else:
        return history[-1]+1
      
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

  def card_estimator_train(self):
    memory_ = np.random.permutation(self.card_estimator_memory)
    memory_idx = range(len(memory_))
    average_loss = 0.0
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
      self.loss_output.write(str(self.count)+','+str(average_loss)+'\n')
      self.loss_output.flush()
  def learn(self,result,my_card,history):
    if self.my_card == None:
      self.my_card = my_card
    self.done=True
  def __del__(self):
    if self.loss_output is not None:
      self.loss_output.close()



def calcsum(cards):
  flags = [c.flag for c in cards]
  cards_num = [c.num for c in cards]
  max_to_0 = (2 in flags)
  summation = sum(cards_num) - max(cards_num) if max_to_0 else sum(cards_num)

  double = (1 in flags)
  summation = 2*summation if double else summation
  return summation

