from card import *
from gaussian import *
from player import *
from DQN import *
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
  train_player = DQNPlayer()
  players.append(train_player)
  players.append(Gaussian(0.3,0.6))#most aggressive
  players.append(Gaussian(0.3,0.7))
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
  print('Evaluation')
  train_player.evaluation_mode = True
  lose_count[:]=0
  for i in range(1000):
    loser = master.play()
    lose_count[loser]+=1
  print(lose_count)
if __name__=='__main__':
  main()
