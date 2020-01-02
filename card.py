
class Card:
    def __init__(self, num, flag):
        self.num = num
        self.flag = flag
    def to_onehot(self):
      if self.flag == 1:
        return 11
      elif self.flag == 2:
        return 12
      elif self.flag == 3:
        return 13
      else:
        if self.num == 20:
          return 0
        elif self.num == 15:
          return 1
        elif self.num == 10:
          return 2
        elif self.num == 5:
          return 3
        elif self.num == 4:
          return 4
        elif self.num == 3:
          return 5
        elif self.num == 2:
          return 6
        elif self.num == 1:
          return 7
        elif self.num == 0:
          return 8
        elif self.num == -5:
          return 9
        elif self.num == -10:
          return 10
    def __str__(self):
        if self.flag == 0:
            return str(self.num)
        elif self.flag == 1:
            return "x2"
        elif self.flag == 2:
            return "max0"
        else:
            return "?"
    def __repr__(self):
        return self.__str__()
    def __eq__(self,other):
        if not isinstance(other,Card):
            return NotImplemented
        else:
            return other.num == self.num  and other.flag == self.flag
