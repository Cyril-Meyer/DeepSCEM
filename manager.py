import data


class Manager:
    def __init__(self):
        self.datasets = dict()

    def get_dataset_index(self):
        return list(self.datasets.keys())

    def load_dataset(self, filename):
        self.datasets[filename] = data.read_dataset(filename)

    def new_dataset(self, name):
        self.datasets[name] = data.create_dataset(name)

    def add_sample(self, dataset_name, sample_name, sample_image, sample_labels):
        dataset = self.datasets[dataset_name]
        data.add_sample_to_dataset(dataset, sample_name, sample_image, sample_labels)

    def remove_dataset(self, filename):
        raise NotImplementedError
        # self.dataset.remove()
