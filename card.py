
class Card:
    def __init__(self, num, flag):
        self.num = num
        self.flag = flag
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
