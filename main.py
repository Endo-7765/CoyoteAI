from card import *
from gaussian import *
from player import *
from DQN import *
from estimator_baysian import *
from multiprocessing import Pool
import itertools
import numpy as np
def defaultCoyoteCards():
  cards = []
  cards.append(Card(20,0))
  cards.append(Card(15,0))
  cards.append(Card(15,0))
  cards.append(Card(10,0))
  cards.append(Card(10,0))
  cards.append(Card(10,0))
  for i in range(4):
    cards.append(Card(5,0))
    cards.append(Card(4,0))
    cards.append(Card(3,0))
    cards.append(Card(2,0))
    cards.append(Card(1,0))
    cards.append(Card(0,0))
  cards.append(Card(-5,0))
  cards.append(Card(-5,0))
  cards.append(Card(-10,0))
  cards.append(Card(0,1))
  cards.append(Card(0,2))
  cards.append(Card(0,3))
  return cards
def generate_player(player_array):
  retval = []
  for i in player_array:
    if i==0:
      retval.append(DQNPlayer())
    elif i==1:
      retval.append(BaysianEstimatorPlayer(0.2,0.8))
    elif i==2:
      retval.append(BaysianEstimatorPlayer(0.4,0.7))
    elif i==3:
      retval.append(Gaussian(0.2,0.8))
    elif i==4:
      retval.append(Gaussian(0.4,0.7))
  return retval
def trial(player_array_seed):
  coyote_cards = defaultCoyoteCards()
  master = GameMaster()
  master.all_cards = coyote_cards
  player_array = np.random.RandomState(seed = player_array_seed).permutation(np.arange(5))
  master.players = generate_player(player_array)
  lose_count = np.zeros(5,dtype=np.int64)
  for i in range(10000):
    loser = master.play()
    lose_count[loser]+=1
    if i%1000 == 0:
      lose_count[:] = 0
      print(i)

  lose_count[:]=0
  for i in master.players:
    if isinstance(i,DQNPlayer):
      i.evaluation_mode=True
  for i in range(10000):
    loser = master.play()
    lose_count[loser]+=1
  retval = np.zeros(5,dtype=np.int64)
  retval[list(player_array)]=lose_count
  print(retval)
  return retval
def main():
  p = Pool()
  result = p.map(trial,list(range(64)))

  print(result)
  print('sum')
  print(np.sum(np.array(result),axis=0))
  
if __name__=='__main__':
  main()
