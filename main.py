from card import *
from gaussian import *
from player import *
from DQN import *
from estimator_baysian import *
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
  players = []
  dqn = DQNPlayer()
  dqn2 = DQNPlayer()
  #players.append(BaysianEstimatorPlayer(0.2,0.8))
  players.append(Gaussian(0.3,0.6))#most aggressive
  players.append(dqn)
  players.append(dqn2)
  players.append(Gaussian(0.2,0.8))
  players.append(Gaussian(0.2,0.9))#most defensive
  
  master.players = players
  lose_count = np.zeros(5,dtype=np.int32)
  for i in range(1000000):
    loser = master.play()
    lose_count[loser]+=1
    if i%1000 == 0:
      print(lose_count)
      lose_count[:] = 0
  lose_count[:]=0
  dqn2.evaluation_mode = True
  dqn1.EPSILON = 1.0
  for i in range(1000000):
    loser = master.play()
    lose_count[loser]+=1
    if i%1000 == 0:
      print(lose_count)
      lose_count[:] = 0

  print('Evaluation')
  lose_count[:]=0
  dqn.evaluation_mode=True
  dqn2.evaluation_mode = True
  for i in range(10000):
    loser = master.play()
    lose_count[loser]+=1
  print(lose_count)
if __name__=='__main__':
  main()
