import csv


class Log_experiences():
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        with open(self.path / 'logs_exps.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames)
            writer.writeheader()

    def add_line(self, dict):
        with open(self.path / 'logs_exps.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, self.fieldnames)
            writer.writerow(dict)

class Log_plots():
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        with open(self.path / 'logs_plots.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames)
            writer.writeheader()

    def add_line(self, dict):
        with open(self.path / 'logs_plots.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, self.fieldnames)
            writer.writerow(dict)


