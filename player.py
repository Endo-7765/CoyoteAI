import random

class Player():
    def __init__(self):
        self.params = {}
        self.history = []
        self.players = []
        self.cards = []
    
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

    def learn(self, result):
        # implement learn
        # parmas = params
        pass
class Human(Player):
  def __call__(self,number):
    # show visible cards to the player
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

    def play(self):
        num_p = len(self.players)
        cards, summation = self.set_cards(random.shuffle(self.all_cards)[0:num_p+1])
        for i, p in enumerate(self.players):
            for c in (cards[:i][::-1] + cards[i:][::-1]):
                p.get_cards(c)
        
        game_flag = True
        history = []
        results = []
        while game_flag:
            num = 0
            for i, p in enumerate(self.players):
                num = p(num)
                if num == -100:
                    game_flag = False
                    if (len(history) == 0 and summation<0) or  (len(history)!=0 and history[-1] > summation):
                        # success!
                        # results = 
                        pass
                    else:
                        pass
                        # failure!
                        # results =
                    break
                else:
                    history.append(num)
        
        self.train(results, history)
        
    def set_cards(self, cards):
        flags = [c.flag for c in cards]
        nums = [c.num for c in cards]

        question = (3 in flags[:-1])
        if question:
            i = flags.index(3)
            nums = nums[:i] + nums[i+1:]
            flags = flags[:i] + flags[i+1]
        else:
            nums = nums[:-1]
            flags = flags[:-1]

        max_to_0 = (2 in flags)
        summation = sum(nums) - max(nums) if max_to_0 else sum(nums)
        
        double = (1 in flags)
        summation = 2*summation if double else summation

        return cards[:-1], sum
    
    def train(self, results, history):
        for p in self.players:
            p.learn(results, history)
