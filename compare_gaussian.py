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
  param1 = np.linspace(0.2,0.7,6)
  param2 = np.linspace(0.5,0.9,5)
  for i in player_array:
    retval.append(Gaussian(param1[i//len(param2)],param2[i%len(param2)]))
  return retval

    
def trial(seed):
  player_array = np.random.RandomState(seed=seed).permutation(np.arange(30,dtype=np.int64))[:5]
  coyote_cards = defaultCoyoteCards()
  master = GameMaster()
  master.all_cards = coyote_cards
  master.players = generate_player(player_array)
  lose_count = np.zeros(5,dtype=np.int64)
  for i in master.players:
    if isinstance(i,DQNPlayer):
      i.evaluation_mode=True
  for i in range(10000):
    loser = master.play()
    lose_count[loser]+=1
  retval = np.zeros(30,dtype=np.float32)
  retval[:]=np.nan
  retval[list(player_array)]=lose_count
  print(retval)
  return retval
def main():
  p = Pool()
  result = p.map(trial,np.arange(1000))

  print(result)
  print('average')
  print(np.nanmean(np.array(result),axis=0))
  
if __name__=='__main__':
  main()
