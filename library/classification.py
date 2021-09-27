import time


class SKLearnClassification:

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    def train(self, train_x, train_y, dataset_name: str):
        self.dataset_name = dataset_name

        start = time.time()
        self.model.fit(train_x, train_y)
        duration = round(time.time() - start, 2)
        print(f"---> execution time : {duration} seconds")

    def predict(self, input_x):
        prediction = self.model.predict(input_x)
        return prediction
