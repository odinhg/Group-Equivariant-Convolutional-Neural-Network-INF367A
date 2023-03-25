class EarlyStopper():
    def __init__(self, limit = 10, delta= 0.01):
        self.limit = limit
        self.delta = delta 
        self.min_acc = 0 
        self.counter = 0
        print(f"Earlystopper active with limit: {self.limit} steps and delta: {self.delta}.")

    def __call__(self, validation_acc):
        if validation_acc > self.min_acc:
            self.min_acc = validation_acc
            self.counter = 0
        elif validation_acc < self.min_acc - self.delta:
            self.counter += 1
            if self.counter >= self.limit:
                return True
        return False
