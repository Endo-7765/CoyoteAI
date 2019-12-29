from player import *
from main import *
from card import *
class Gaussian(Player):
  def __init__(self,defensive_rate,coyote_rate):
    super(Gaussian,self).__init__()
    self.defensive_rate = defensive_rate
    self.coyote_rate = coyote_rate

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
    threshold.sort()
    if len(history)==0:#自分が初めての手番の場合
      if threshold[int(self.coyote_rate * len(threshold))] < 0:#coyote_rate以上の確率で、真値が0以下と確信
        return -100
      elif threshold[int(self.defensive_rate * len(threshold))] < 0: #defensive_rateでは0以下だが、coyote_rateでは0以上のとき、0と答える
        return 0
      else:#それ以外のときは、defensive_rateの値を答える
        return threshold[int(self.defensive_rate * len(threshold))]
    else:
      if history[-1] >= threshold[int(self.coyote_rate * len(threshold))]:#前の人の言った値が、coyote_rate以上の確率で真値オーバー
        return -100#コヨーテ
      elif history[-1]>=threshold[int(self.defensive_rate * len(threshold))]:#前の人の言った値が、defensive_rate以上
        return history[-1]+1
      else:
        return threshold[int(self.defensive_rate * len(threshold))]
def calcsum(cards):
  flags = [c.flag for c in cards]
  cards_num = [c.num for c in cards]
  max_to_0 = (2 in flags)
  summation = sum(cards_num) - max(cards_num) if max_to_0 else sum(cards_num)

  double = (1 in flags)
  summation = 2*summation if double else summation
  return summation
