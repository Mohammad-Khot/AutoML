# core/state.py

class AutoMLState:
    def __init__(self):
        self.scores = {}

    def update(self, model_name, score):
        self.scores[model_name] = score
