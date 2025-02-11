import random
import unittest
import csv
import numpy as np
from AuditorController import AuditorController
from LLMTools import parse_command_map, parse_test_input_map, get_embedding, get_command_embedding_map, \
    get_input_variant_embedding_map
from ModelAssessor import ModelAssessor


class MyTestCase(unittest.TestCase):
    def setUp(self):
        print("Starting tests\n\n*******")

    def test_can_construct_option_string_from_list(self):
        expected_sentiment = "A task was accomplished successfully"
        incorrect_sentiment = "There was an error executing the task"
        options = [expected_sentiment, incorrect_sentiment]

        quote_items = False
        include_header_footer = True
        if include_header_footer:
            out = "---------------\n"
        else:
            out = ""
        for i, option_desc in enumerate(options):
            if quote_items:
                options = f"({i+1}) \"{option_desc}\"\n"
            else:
                options = f"({i+1}) {option_desc}\n"
            out += options
        if include_header_footer:
            out += "---------------\n"
        print(f"\n{out}")


    def test_sentiment_auditor_model(self):
        auditor_model_name = "llama3"
        auditor_model_name = "deepseek-small"

        auditor = AuditorController(auditor_model_name)

        input_under_assessment = "Succeeded in creating new folder"

        expected_sentiment = "A task was accomplished successfully"
        incorrect_sentiment = "There was an error executing the task"
        options = [expected_sentiment, incorrect_sentiment]
        self.assertTrue(expected_sentiment == auditor.getClosestSentiment(input_under_assessment,options))

    def test_llm_output_cleaner(self):
        auditor_model_name = "llama3"

        auditor = AuditorController(auditor_model_name)

        verbose_output = """
        <think>
Okay, so I need to figure out which of the two options (1 or 2) best matches the sentiment of the given input sentence. The input sentence is "Succeeded in creating new folder." Let me break this down.

First, the input sentence is positive because it says that a task was accomplished successfully—specifically, creating a new folder. That's a clear success, so the sentiment here is positive or successful completion.

Now looking at the options:

1) A task was accomplished successfully
2) There was an error executing the task

Option 1 directly states success, which aligns perfectly with the input sentence. The input mentions success in creating a folder, and option 1 is about a task being accomplished successfully without any negative connotation.

Option 2 talks about an error, which would be negative or unsuccessful. Since the input doesn't mention any errors but rather a successful action, option 2 doesn't fit here.

Therefore, the most similar option to the input sentence's sentiment is option 1.
</think>

The input sentence expresses success, matching option (1).

1
        """
        deVerbose = auditor.deverbose(verbose_output)
        self.assertTrue("1" == deVerbose)


    def test_can_retrieve_system_command_templates(self):
        template_file = "command_index.csv"

        # Only testing enough to validate algorithm, will need to add more
        # test keys if we want this test to also validate the particular set of commands and their expected
        # variants
        expected_keys = [("execute_command", 2), ("change_models", 1)]

        command_map = parse_command_map(template_file)
        self.assertTrue(all([ _[0] in command_map and _[1] == len(command_map[_[0]]) for _ in expected_keys]))


    def test_can_retrieve_test_input_variants(self):
        variants_file = "command_test_inputs.csv"
        expected_keys = [("execute_command", 5), ("create_branch", 3)]

        test_eg_map = parse_test_input_map(variants_file)

        self.assertTrue(all([ _[0] in test_eg_map and _[1] == len(test_eg_map[_[0]]) for _ in expected_keys]))


    def test_can_read_command_template(self):
        template_file = "command_index.csv"
        data = csv.reader(open(template_file))
        command_specs = list(data)[1:]
        print(f"Out: ${command_specs}")

        out = {}

        for command_type, com_variant in command_specs:
            if command_type in out:
                out[command_type].append(com_variant)
            else:
                out[command_type] = [com_variant.strip()]
        print(f"\n\n{out}")

    def test_can_create_model_assessor(self):
        model_name, url, api_key, min_fract = ("mxbai-embed-large:latest", "http://localhost:11434/v1/", "ollama", 0.0)
        assessor = ModelAssessor(model_name, url, api_key, min_fract)


    def test_can_compute_embedding_for_input(self):
        expected_emb_length = 1024 # This number comes from `ollama show "mxbai-embed-large:latest"`
        model_name = "mxbai-embed-large:latest"
        model_name, url, api_key = (model_name, "http://localhost:11434/v1/", "ollama")
        template_file = "command_index.csv"
        command_map = parse_command_map(template_file)
        self.assertTrue(len(command_map) > 0)
        command_key = list(command_map)[0]
        variants = command_map[command_key]
        self.assertTrue(len(variants) > 0)
        template_variant = variants[0]
        print(f"Getting embedding for {command_key}: {template_variant}")


        embedding_vec = get_embedding(template_variant, model_name, url, api_key)

        self.assertTrue(len(embedding_vec) == expected_emb_length)

    def test_can_compute_command_embedding_map(self):
        expected_emb_length = 1024 # This number comes from `ollama show "mxbai-embed-large:latest"`
        model_name = "mxbai-embed-large:latest"
        model_name, url, api_key = (model_name, "http://localhost:11434/v1/", "ollama")
        template_file = "command_index.csv"
        command_map = parse_command_map(template_file)

        embedding_map = get_command_embedding_map(command_map, model_name, url, api_key)

        self.assertTrue(len(embedding_map) > 0)
        self.assertTrue(all([len(_) == expected_emb_length for _ in embedding_map.values()]))

    def test_can_compute_input_variant_embedding_map(self):
        expected_emb_length = 1024  # This number comes from `ollama show "mxbai-embed-large:latest"`
        model_name = "mxbai-embed-large:latest"
        model_name, url, api_key = (model_name, "http://localhost:11434/v1/", "ollama")
        variants_file = "command_test_inputs.csv"
        expected_keys = [("execute_command", 5), ("create_branch", 3)]

        test_eg_map = parse_test_input_map(variants_file)

        self.assertTrue(all([ _[0] in test_eg_map and _[1] == len(test_eg_map[_[0]]) for _ in expected_keys]))



        embedding_map = get_input_variant_embedding_map(test_eg_map, model_name, url, api_key)

        self.assertTrue(len(embedding_map) > 0)

        variants = list(embedding_map)
        com_type = variants[0]
        first = embedding_map[com_type]
        variant = first[0]
        self.assertTrue(len(variant) == expected_emb_length)

        self.assertTrue(all([(variant in embedding_map and num_examples == len(embedding_map[variant]) and expected_emb_length == len(embedding_map[variant][0])) for (variant, num_examples) in expected_keys]))

    def test_can_find_closest_command_template_to_input(self):
        model_name = "mxbai-embed-large:latest"
        model_name, url, api_key = (model_name, "http://localhost:11434/v1/", "ollama")
        template_file = "command_index.csv"
        test_inputs_files = "command_test_inputs.csv"

        command_map = parse_command_map(template_file)
        inputs_map = parse_test_input_map(test_inputs_files)

        command_embedding_map = get_command_embedding_map(command_map, model_name, url, api_key)
        test_input_emb_map = get_input_variant_embedding_map(inputs_map, model_name, url, api_key)

        example_type = "create_branch"
        choice_index = random.choice(range(len(inputs_map[example_type])))
        example_input = inputs_map[example_type][choice_index]
        input_embedding = test_input_emb_map[example_type][choice_index]
        print(f"Find template most similar to: {example_input}")

        all_command_items = list(command_embedding_map.items())
        all_command_embeddings = np.array([_[1] for _ in all_command_items])
        all_command_types = [_[0] for _ in all_command_items]

        print(f"embeddings shape: {all_command_embeddings.shape}")

        inp = all_command_embeddings - input_embedding.reshape(1, 1024)

        norms = np.linalg.norm(inp, 2, axis=1)

        closest_arg = np.argmin(norms )
        closest_type = all_command_types[closest_arg]
        print(f"Closest template type is {closest_type}")
        self.assertTrue(closest_type == example_type)


    def test_can_assess_embedding_model(self):
        min_fract = 1.0
        model_name = "mxbai-embed-large:latest"
        model_name, url, api_key, min_fract = (model_name, "http://localhost:11434/v1/", "ollama", min_fract)
        assessor = ModelAssessor(model_name, url, api_key, min_fract)

        result = assessor.assessEmbeddingModel()
        did_pass, score = result
        print(f"Test results: {result}")
        self.assertTrue(did_pass and score >= min_fract)










if __name__ == '__main__':
    unittest.main()
