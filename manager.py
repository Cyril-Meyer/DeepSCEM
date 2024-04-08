import data


class Manager:
    def __init__(self):
        self.datasets = []

    def load_dataset(self, filename):
        self.add_dataset(data.read_dataset(filename))

    def add_dataset(self, dataset):
        self.datasets.append(dataset)

    def remove_dataset(self, filename):
        raise NotImplementedError
        # self.dataset.remove()
