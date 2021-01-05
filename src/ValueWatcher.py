import sys


class ValueWatcher:
    def __init__(self, mode='maximize', threshold=5):
        self.current = None
        self.count = 0
        self.max_score = sys.float_info.min
        self.min_score = sys.float_info.max
        self.is_max = False
        self.is_min = False
        self.is_new = False
        self.mode = mode
        self.threshold = threshold

    def update(self, val):
        if self.current is None:
            self.current = val
            self.max_score = val
            self.min_score = val
        else:
            self.current = val
            self.is_new = False
            if self.mode == 'maximize':
                if val > self.max_score:
                    self.count = 0
                    self.max_score = val
                    self.is_new = True
                else:
                    self.count += 1
            elif self.mode == 'minimize':
                if val < self.min_score:
                    self.count = 0
                    self.min_score = val
                    self.is_new = True
                else:
                    self.count += 1

        if self.count >= self.threshold:
            if self.mode == 'maximize':
                self.is_max = True
            elif self.mode == 'minimize':
                self.is_min = True
        else:
            self.is_max = False
            self.is_min = False

    def is_over(self):
        if self.mode == 'maximize':
            return self.is_max
        elif self.mode == 'minimize':
            return self.is_min
        else:
            return None

    def is_updated(self):
        return self.is_new