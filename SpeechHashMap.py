from LLMTools import EMBEDDING_MODEL_NAME, get_embedding
import numpy as np
class SpeechHashMap:
    def __init__(self, data_dict = None, auto_fill = None ):
        if data_dict is None:
            data_dict = dict()
        if auto_fill is None:
            auto_fill = True
        self.data_dict = data_dict
        self.embedding_map = dict()
        self.embedding_model = EMBEDDING_MODEL_NAME
        if auto_fill:
            self.fill()

    def fill(self):
        for key in self.data_dict:
            if key not in self.embedding_map:
                self.embedding_map[key] = get_embedding(key, model=self.embedding_model)

    def set(self, key, value):
        self.data_dict[key] = value
        self.fill()
        return value

    def get(self, key):
        if key in self.data_dict:
            return self.data_dict[key]
        else:
            return None

    def find(self, query, max_results = None):
        """
        Returns a list of the top keys that are a fuzzy match for query.  Returns empty list if no match
        """
        if max_results is None:
            max_results = 1
        from operator import itemgetter
        query_embedding = get_embedding(query, model=self.embedding_model)
        dists = [(key_embedding[0], np.linalg.norm(query_embedding - key_embedding[1])) for key_embedding in self.embedding_map.items()]
        sorted_dists = sorted(dists, key=itemgetter(1))
        return [key[0] for key in sorted_dists[:max_results]]