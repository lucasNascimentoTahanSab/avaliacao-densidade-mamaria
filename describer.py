import math


class Describer:

    def __init__(self, co_occurence_matrix):
        self.co_occurrence_matrix = co_occurence_matrix
        self.entropy = 0

    def get_entropy(self):
        for co_occurence_row in self.co_occurence_matrix:
            for co_occurrence_node in co_occurence_row:
                self.entropy += co_occurrence_node * math.log(co_occurrence_node, 2)

        self.entropy = -self.entropy

        return self.entropy