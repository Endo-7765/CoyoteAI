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

def main():
  coyote_cards = defaultCoyoteCards()
  master = GameMaster()
  master.all_cards = coyote_cards
  players= []
  baysian1 = BaysianEstimatorPlayer(0.2,0.8,loss_output = "baysian1_no_DQN.log")
  baysian2 = BaysianEstimatorPlayer(0.3,0.7,loss_output = "baysian2_no_DQN.log")
  baysian3 = BaysianEstimatorPlayer(0.4,0.7,loss_output = "baysian3_no_DQN.log")
  players.append(baysian1)
  players.append(Gaussian(0.3,0.6))
  players.append(baysian2)
  players.append(Gaussian(0.2,0.8))
  players.append(baysian3)

  master.players = players
  lose_count = np.zeros(5,dtype=np.int64)
  for i in range(100000):
    loser = master.play()
    lose_count[loser]+=1
    if i%1000 == 0:
      print(lose_count)
      lose_count[:] = 0
  lose_count[:]=0
  
 
  for i in range(10000):
    loser = master.play()
    lose_count[loser]+=1
  print(lose_count)
if __name__=='__main__':
  main()
