from player import *
from main import *
from card import *
class Gaussian(Player):
  def __init__(self,aggressive_rate):
    super(Gaussian,self).__init__()
    self.aggressive_rate = aggressive_rate

  def __call__(self,history):
    cards = defaultCoyoteCards()
    for i in self.cards:
      cards.remove(i)
    #The rest cards are candidate of my card
    flags = [c.flag for c in cards]
    if (3 in flags):
      cards.remove(Card(0,3))
    threshold = []
    for i in cards:
      threshold.append(calcsum(self.cards + [i]))
    threshold.sort(reverse=True)
    print(threshold)
    if len(history)==0:
      if threshold[int(self.aggressive_rate * len(threshold))] < 0:
        return -100
      else:
        return threshold[int(self.aggressive_rate * len(threshold))]
    else:
      if history[-1] < threshold[int(self.aggressive_rate * len(threshold))]:
        return threshold[int(self.aggressive_rate * len(threshold))]
      else:
        return -100
def calcsum(cards):
  flags = [c.flag for c in cards]
  cards_num = [c.num for c in cards]
  max_to_0 = (2 in flags)
  summation = sum(cards_num) - max(cards_num) if max_to_0 else sum(cards_num)

  double = (1 in flags)
  summation = 2*summation if double else summation
  return summation
