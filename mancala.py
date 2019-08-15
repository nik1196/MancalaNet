import random
class Mancala:
    def __init__(self):
        self.pits = [5 for i in range(14)]
        random.seed()
        
    def play(self, pit_num, gain=0):  
        pit = pit_num - 1
        next_pit = (pit + self.pits[pit]+1) % 14
        num_iter = self.pits[pit]
        self.pits[pit] = 0
        for i in range(pit+1, pit+num_iter+1):
            self.pits[i%14] += 1
        if self.pits[next_pit] > 0:
            return gain + self.play(next_pit+1, gain)
        score = self.pits[(next_pit+1) % 14]
        self.pits[(next_pit+1) % 14] = 0
        return score

    def get_best_pit(self, as_one_hot = True):
        max_score = 0
        best_pit = 0
        for pit in range(6,len(self.pits)):
            if self.pits[pit] == 0:
                continue
            pits_temp = list(self.pits)
            score = self.play(pit+1)
            if score > max_score:
                max_score = score
                best_pit = pit
            self.pits = list(pits_temp)
        if as_one_hot == False:
            return best_pit+1
        return [ 1 if i == best_pit else 0 for i in range(len(self.pits))]

    
    def generate_state(self):
        total_seeds = 70
        for i in range(14):
            self.pits[i] = random.randrange(0,total_seeds+1)
            total_seeds -= self.pits[i]
        return self.pits

    def get_pits(self):
        return self.pits

    def reset(self):
        self.pits = [5 for i in range(14)]
        
