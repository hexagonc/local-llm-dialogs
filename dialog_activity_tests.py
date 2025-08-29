import unittest
from hive.DialogActivity import DialogActivity
from LLMTools import flatten_dialog_list, parse_roles_from_dialog_pattern_file
from hive.Hive import Hive
from LLMChatConfig import CONFIG_MAP

class MyTestCase(unittest.TestCase):

    def setUp(self):
        print("\n")
    def test_can_create_new_Hive(self):
        hive = Hive()

    def test_can_chat_with_Hive(self):
        hive = Hive()
        response = hive.user_chat("What day is it?")
        print("Basic hive response\n")
        print(response)

    def test_can_add_a_dialog_activity_to_hive(self):
        hive = Hive()
        topic = "discussions about alarms that I have created or plan to create"
        response = hive.add_dialog_activity(topic)
        print(response)

    def test_can_search_for_existing_activity(self):
        hive = Hive()

        topic = "discussions about alarms that I have created or plan to create"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)
        topic = "grocery list"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)
        topic = "reminders"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)

        not_found_key = "not found"
        topic_to_find = "to do list"
        response = hive.find_existing_topic(topic_to_find, not_found_key)
        print(f"Found topic: \"{topic_to_find}\"?")
        print(response)

        if not response == not_found_key:
            print("Expected new topic to be created.  Force create new")
            response = hive.add_dialog_activity(topic_to_find)
            print(f"New topic response:\n{response}")

        topic_to_find = "alarms"
        print(f"Found topic: \"{topic_to_find}\"?")
        response = hive.find_existing_topic(topic_to_find)
        print(response)


    def test_automatic_dialog_dispatching_via_hive(self):
        hive = Hive()

        topic = "discussions about alarms that I have created or plan to create"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)
        topic = "all about my grocery list, including reading, adding and removing items"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)
        topic = "reminders"
        print(f"Creating dialog activity from: \"{topic}\":")
        response = hive.add_dialog_activity(topic)
        print(response)

        command = "add the item 'buy bananas'"
        response = hive.user_chat(command)
        print(f"Response to {command}\n")
        print(response)



    def test_can_create_dialog_activity(self):
        topic = "root conversation about nothing in particular"
        topic_short_name = "root_conversation_context"
        topic_meta_role_name = "base-dialog"
        hive = Hive()
        dialog_activity = DialogActivity(hive, topic, topic_short_name, topic_meta_role_name)

    def test_can_write_logs(self):
        from hive.NoeticLogger import NoeticLogger
        logger = NoeticLogger("test_hive_logs")
        logger.logDebug("o___o", "Testing", verbose=True)

    def test_speaking_to_dialog_activity_extends_knowledge(self):
        hive = Hive()

        topic = "all about my grocery list, including reading, adding and removing items"
        print(f"Creating dialog activity from: \"{topic}\":")
        activity, role = hive.add_dialog_activity(topic)
        activity:DialogActivity
        role:str
        self.assertTrue(activity)
        resp = activity.user_chat("Add chilli")
        print(f"Response: {resp}" )

        print(hive.__class__.__name__+ "\n")

    def test_can_rule_lisp_code(self):
        hive = Hive()

        topic = "Use plisp to show the current date and time."
        print(f"AI to call lisp")
        resp = hive.user_chat(topic)
        print(f"Done:\n{resp}")





if __name__ == '__main__':
    unittest.main()
