from LLMTools import parse_command_map, parse_test_input_map, get_command_embedding_map, get_input_variant_embedding_map
import numpy as np

class ModelAssessor:
    def __init__(self, model_name, url, api_key, min_pass_score = None):
        if min_pass_score is None:
            min_pass_score = 0.0
        template_file = "command_index.csv"
        test_inputs_files = "command_test_inputs.csv"
        self.command_template_map = parse_command_map(template_file)
        self.input_example_map = parse_test_input_map(test_inputs_files)
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        self.min_passing_score = min_pass_score


    def assessEmbeddingModel(self):
        model_name, url, api_key = (self.model_name, self.url, self.api_key)
        command_embedding_map = get_command_embedding_map(self.command_template_map, self.model_name, self.url, self.api_key)
        test_input_emb_map = get_input_variant_embedding_map(self.input_example_map, model_name, url, api_key)

        success_count = 0.0
        num_tests = 0.0

        all_command_items = list(command_embedding_map.items())
        all_command_embeddings = np.array([_[1] for _ in all_command_items])
        all_command_types = [_[0] for _ in all_command_items]

        for command_type, test_input_embeddings in test_input_emb_map.items():
            for input_embedding in test_input_embeddings:
                num_tests += 1
                emb_length = len(input_embedding)
                inp = all_command_embeddings - input_embedding.reshape(1, emb_length)
                norms = np.linalg.norm(inp, 2, axis=1)

                closest_arg = np.argmin(norms)
                closest_type = all_command_types[closest_arg]
                if closest_type == command_type:
                    success_count +=1.0

        success_fract = success_count/num_tests

        return success_fract >=self.min_passing_score, success_fract