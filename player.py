import random
import numpy as np
class Player():
    def __init__(self):
        self.params = {}
        self.history = []
        self.players = []
        self.cards = []
    def reset(self):
        self.cards=[]
    def get_cards(self, card):
        self.cards.append(card)

    def judge(self, number):
        # implement judge
        is_coyote = (random.random() < 0.2)
        num = number + 1
        return num, is_coyote

    def __call__(self, number):
        num, is_coyote = self.judge(number)
        if is_coyote:
            return -100
        else:
            return num

    def learn(self, result,history):
        # implement learn
        # parmas = params
        pass
class Human(Player):
  def __call__(self,history):
    # show visible cards to the player
    print('history')
    print(history)
    print(self.cards)
    num = ""
    while True:
      num = input()
      try:
        num = int(num)
        break
      except ValueError:
        print('wrong input')
    if num<=-1:
      num = -100
    return num

class GameMaster():
    def __init__(self):
        self.players = []
        self.all_cards = []
        self.player_index = 0

    def play(self):
        loser = 0
        num_p = len(self.players)
        random.shuffle(self.all_cards)
        cards, summation = self.set_cards(self.all_cards[0:num_p+1])
        for i, p in enumerate(self.players):
            p.reset()
            for c in (cards[:i][::-1] + cards[i+1:][::-1]):
                p.get_cards(c)
        
        game_flag = True
        history = []
        results = []
        while game_flag:
            p = self.players[self.player_index]
            num = p(history)
            if num == -100:
                #print("player"+str(self.player_index)+"replyed coyote")
                game_flag = False
                #print("true cards")
                #print(cards)
                #print(summation)

                if (len(history) == 0 and summation<0) or  (len(history)!=0 and history[-1] > summation):
                    # success!
                    results = np.ones(len(self.players),dtype=np.float32) * 0.25
                    results[(self.player_index-1)%len(self.players)] = -1
                    loser = (self.player_index-1)%len(self.players)
                else:
                    # failure!
                    results = np.ones(len(self.players),dtype=np.float32)*0.25
                    results[self.player_index]=-1
                    loser = self.player_index
                break
            elif (len(history)>0 and num > history[-1]) or (len(history)==0 and num >=0):
                history.append(num)
                #print("player"+str(self.player_index)+"replyed"+str(num))
            else:
                print('illegal number from'+str(self.player_index))
                print('history')
                print(history)
                print('reply')
                print(num)
            self.player_index = (self.player_index+1)%len(self.players)
        
        self.train(results, history)
        return loser
        
    def set_cards(self, cards):
        flags = [c.flag for c in cards]
        nums = [c.num for c in cards]

        question = (3 in flags[:-1])
        if question:
            i = flags.index(3)
            nums = nums[:i] + nums[i+1:]
            flags = flags[:i] + flags[i+1:]
        else:
            nums = nums[:-1]
            flags = flags[:-1]

        max_to_0 = (2 in flags)
        summation = sum(nums) - max(nums) if max_to_0 else sum(nums)
        
        double = (1 in flags)
        summation = 2*summation if double else summation

        return cards[:-1], summation
    
    def train(self, results, history):
        for i in range(len(self.players)):
          self.players[i].learn(results[i],history)
