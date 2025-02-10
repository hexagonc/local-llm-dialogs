import unittest

from AuditorController import AuditorController


class MyTestCase(unittest.TestCase):

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

        auditor = AuditorController(auditor_model_name)

        input_under_assessment = "Succeeded in creating new folder"

        expected_sentiment = "A task was accomplished successfully"
        incorrect_sentiment = "There was an error executing the task"
        options = [expected_sentiment, incorrect_sentiment]
        self.assertTrue(expected_sentiment == auditor.getClosestSentiment(input_under_assessment,options))




if __name__ == '__main__':
    unittest.main()
