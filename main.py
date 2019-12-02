from card import *
from gaussian import *
from player import *
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
  players.append(Human())
  players.append(Gaussian(0.4))
  players.append(Gaussian(0.5))
  players.append(Gaussian(0.6))
  players.append(Gaussian(0.7))
  
  master.players = players
  master.play()
if __name__=='__main__':
  main()
