import unittest
import os
from LLMTools import read_prompt_file, get_delimited_text, do_multi_shot_llm_query
from plisp.Environment import Environment

from plisp.LispTools import add_basic_functions
from plisp.LispTools import add_arithmetic_functions, LispTools
from hive.HiveUtils import build_prompt_from_template
from plisp.Value import Value
from LLMChatConfig import CONFIG_MAP


def to_svalue(s) -> str:
    return LispTools.make_str(s)
class MyTestCase(unittest.TestCase):

    def setUp(self):
        print("\n\n")
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_can_read_lisp_activity_file(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)

        plisp_definition_prompt_file = "/Users/evolvedgpt/Dropbox/Apps/AndroidLispGUIBuilder/speechbot_5/jlisp_language_dialog_def.txt"

        env.map_value("plisp_file",  to_svalue(read_prompt_file(plisp_definition_prompt_file)))
        dir = os.getcwd()
        activity_name_base_file_name = "get_lisp.txt"
        activity_file_abs_path = f"{dir}/{activity_name_base_file_name}"

        raw = read_prompt_file(activity_file_abs_path)
        print("Raw prompt:")
        print(raw)
        start_delimiter = "{{"
        end_delimiter = "}}"
        replacements = get_delimited_text(raw, start_delimiter, end_delimiter)
        if len(replacements) > 0:
            for field in replacements:
                lisp_exp = field[0]
                res:Value = env.evaluate_exp(lisp_exp)
                raw = raw.replace(f"{start_delimiter}{lisp_exp}{end_delimiter}", res.string())
        print("Full prompt:\n")
        print(raw)

    def test_can_replace_field_in_prompt_template(self):
        env = Environment()
        add_basic_functions(env)

        env.map_value("today", LispTools.make_str("10/01/1979"))
        prompt_template = "Consider the current date: {{today}}, return the function that formats this date in long form, '{long month name} {day of month, zero padded for single digit days}, {year}, {weekday}'"
        new = build_prompt_from_template(env, prompt_template)
        print("Actual prompt:\n")
        print(new)



    def test_can_use_a_model_to_accomplish_a_goal(self):
        from LLMTools import user_prompt_segment, assistant_prompt_segment
        env = Environment()
        add_basic_functions(env)
        client_role_prefix = "calculator"
        client_descr = f"I am a smart calculator app that understands English language but doesn't know Lisp.  I need you to translate a user's instructions into the appropriate lisp code.  I will prefix any instructions from the user with the role prefix: {client_role_prefix} followed by a colon."
        client_model_descr = "activity-service-client"

        env.map_value("client-prefix", to_svalue(client_role_prefix))
        env.map_value(client_model_descr, to_svalue(client_descr))

        plisp_definition_prompt_file = "/Users/evolvedgpt/Dropbox/Apps/AndroidLispGUIBuilder/speechbot_5/jlisp_language_dialog_def.txt"

        abridged_ = "/Users/evolvedgpt/development/projects/local-llm-dialogs/plisp_language_dialog_def.txt"
        env.map_value("plisp_file", LispTools.make_str(read_prompt_file(abridged_)))
        dir = os.getcwd()
        activity_name_base_file_name = "get_lisp.txt"
        activity_file_abs_path = f"{dir}/{activity_name_base_file_name}"

        activity_template_prompt = read_prompt_file(activity_file_abs_path)

        activity_purpose = build_prompt_from_template(env, activity_template_prompt)
        #print(activity_purpose)

        client_command = f"{client_role_prefix}: Save the value 23 to the variable 'x'.  Then print the result.  Return a single lisp command for this within a progn."
        history = []
        llm_response = do_llm_query(history, activity_purpose)
        history.append(user_prompt_segment(activity_purpose))
        print("LLM response (should be acknowledgement)\n")
        print(llm_response) # Should
        history.append(assistant_prompt_segment(llm_response))

        llm_response = do_llm_query(history, client_command)
        history.append(user_prompt_segment(client_command))

        history.append(assistant_prompt_segment(llm_response))

        if (llm_response.startswith(f"{client_role_prefix}:")):
            print("Correct initial response: Score 4+\n")
            print(llm_response)
            follow_up_query = f"{client_role_prefix}: Now, assume that the variable y contains a list of strings.  Assume there is lisp function called (tts {{string}}) which will speak aloud its string argument.  Show the code for looping over each string in y, tts'ing each value"
            llm_response = do_llm_query(history,follow_up_query)
            history.append(user_prompt_segment(follow_up_query))
            history.append(assistant_prompt_segment(llm_response))
            if (llm_response.startswith(f"{client_role_prefix}:")):
                print("Complete success.  All tests passed, score 5:\n")
                print(llm_response)
            else:
                correction_user_input = "No, use the convention I described to you for responding to the client using the correct response prefix.  Do not apologize. Just return the correct response"
                llm_response = do_llm_query(history, correction_user_input)
                history.append(user_prompt_segment(correction_user_input))
                history.append(assistant_prompt_segment(llm_response))
                if (llm_response.startswith(f"{client_role_prefix}:")):
                    print("Mixed success, required occasional prompting to get correct format, score 4.5\n")
                else:
                    print("Failed to get correct answer even after corrective prompting.  Score 0\n")
                print(llm_response)
        else:
            correction_user_input = "No, use the convention I described to you for responding to the client using the correct response prefix.  Do not apologize. Just return the correct response"
            llm_response = do_llm_query(history, correction_user_input)
            history.append(user_prompt_segment(correction_user_input))
            history.append(assistant_prompt_segment(llm_response))

            if (llm_response.startswith(f"{client_role_prefix}:")):
                # Worked first time after prompting
                print(llm_response)
                llm_response = do_llm_query(history, f"{client_role_prefix}: Now, assume that the variable y contains a list of strings.  Assume there is lisp function called (tts {{string}}) which will speak aloud its string argument.  Show the code for looping over each string in y, tts'ing each value")
                history.append(assistant_prompt_segment(llm_response))
                if (llm_response.startswith(f"{client_role_prefix}:")):
                    print("Semi-success response (got correct future answers), score 4:\n")
                    print(llm_response)
                else:
                    correction_user_input = "No, use the convention I described to you for responding to the client using the correct response prefix.  Do not apologize. Just return the correct response"
                    llm_response = do_llm_query(history, correction_user_input)
                    history.append(user_prompt_segment(correction_user_input))
                    history.append(assistant_prompt_segment(llm_response))
                    if (llm_response.startswith(f"{client_role_prefix}:")):
                        print("Partial success, required repeated prompting, score 3\n")
                    else:
                        print("Failed to get correct answer even after repeated corrective prompting.  Score 0:\n")
                    print(llm_response)
            else:
                print("Failed to get correct response event with prompting.  Score is 0:\n")
                print(llm_response)

    def test_can_create_dialog_activity(self):
        from hive.DialogActivity import DialogActivity
        config_file = "dialogs/dialog_root_conversation.json"
        dialog_activity = DialogActivity(config_file)

    def test_can_serialize_deserialize_env(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)

        test_values = [("date", LispTools.make_str("5/15/2025 9:13 PM")), ("x", LispTools.make_integer(10)), ("y", LispTools.make_integer(100)), ("name", LispTools.make_str("top dialog"))]

        for test_pair in test_values:
            var_name, var_value = test_pair
            env.map_value(var_name, var_value)

        string_hashtable = env.evaluate_exp("(make-string-hashtable (list (list \"x\" 10) (list \"y\" 20)))")
        env.map_value("table", string_hashtable)
        env.map_value("key", to_svalue("Fifty"))
        env.evaluate_exp("(defhash table \"z\" key)")
        serialize = env.serialize()

        new_env = Environment()
        new_env.fromSerialized(serialize)
        d = new_env.get_value("date")
        self.assertEquals(d.string(), "5/15/2025 9:13 PM")
        tvalue = new_env.get_value("table")
        print(f"Table:\n{tvalue}")

    def test_can_parse_out_dialog_sequence(self):
        from LLMTools import parse_roles_from_dialog_pattern_file, flatten_dialog_list, parse_roles_from_dialog_string
        dialog_seq_raw = """
        system: Your job is to tell me the time at all instances.  When you
        finish execute the function (done)
        user: What time is it?
        assistant: It is friday
                """

        dialog = parse_roles_from_dialog_string(dialog_seq_raw, True)
        print(dialog)


    def test_can_parse_gated_prompt_segments(self):
        dialog_env = Environment()
        add_basic_functions(dialog_env)
        add_arithmetic_functions(dialog_env)

        map_svalue(dialog_env, "topic", "top level conversation context about nothing in particular")

    def test_can_describe_common_prompt(self):
        dialog_env = Environment()
        add_basic_functions(dialog_env)
        add_arithmetic_functions(dialog_env)

        map_svalue(dialog_env, "end-of-message-key", "bye")
        map_svalue(dialog_env, "to-do-role", "to-do")
        topic = "keeps track of things that the user wants to get done and is doing"
        map_svalue(dialog_env, "to-do-topic", topic)
        dialog_env.map_value("other-dialog-managers", dialog_env.evaluate_exp(
            f"(list (list to-do-role to-do-topic))"))
        map_svalue(dialog_env, "my-role", "top-dialog")
        map_svalue(dialog_env, "lisp-interpreter-role-name", "slisp")
        map_svalue(dialog_env, "lisp-name", "slisp")
        lisp_language_api_description = """
        This Lisp runtime environment is very simple and only supports a small subset of the functionality of Common Lisp.  Thus, I will refer to this Lisp dialect as
        "{{lisp-name}}" for Simple Lisp.
        ### Values
In {{lisp-name}}, a 'Value' refers to any object instance that can be bound to variables and passed to functions. All Values can be evaluated, transforming them into potentially different Values. {{lisp-name}} categorizes Values into basic and complex types:

- **Basic Types**: Evaluate to themselves and include:
  - **Scalar**: 64-bit integers, 64-bit doubles, strings, booleans (all values are `Truthy` except the singleton `F`) and symbols (unescaped strings).
  - **Composite**: Hashtables with string or integer keys.

- **Complex Types**: May not return verbatim when evaluated:
  - **List**: Delimited by `()`. The first element determines if it's a function/macro call or returns the list itself.
### Functions
{{lisp-name}} only supports a small subset of standard Common Lisp functions as described in the following table:
| Function | Purpose | Example usages |
| --- | --- | --- |
| (setq {variable-name} {value}) | use to assign a value {value} to {variable-name} | (setq x 12) (setq y "twelve") |
| (concat {arg:string}+) | used to concatenate the string arg values into a single string. | (concat "y: " y) (concat "x = " (string x)) |
| (string {arg}) | converts {arg} into a string. | (concat "x = " (string "12")) |
| (list {arg}*) | constructs a list from the arguments passed into the function | (list 1 23 45) |
| (nth {base-list} {index}) | returns the item in position {index} of the list. An exception is thrown if {index} is invalid. | (setq items (list "yes" "no" "maybe")) (nth items 0) -> "yes" |
| (set-nth {base-list} {index} {value}) | destructively updates the value of {base-list} at index {index} to {value}. | (setq items (list "yes" "no" "maybe")) (set-nth items 0 "affirmative") -> ("affirmative" "no" "maybe") |
| (length {composite-value}) | returns the number of elements in {composite-value}. | (length (make-int-hashtable (list (1 "one") (2 "two") (3 "three")))) -> 3 (length (list 1 3 4 5 5)) -> 5 |
| (progn {body-exp}+) | executes each {body-exp} sequentially and returns the value of the last expression. | (progn (setq x 10) (setq y 20)) -> 20 (if (= 12 12) (progn (setq z 1) (setq x (+ x z)))) -> 11 |
| (append-item {arg:list} {item}) | appends {item} to the end of {arg} | (setq foods ()) (setq foods (append-item foods "milk")) (setq foods (append-item foods "water")) |
| (unbind {var-name:symbol}) | Unbinds the {var-name} in the outermost environment within scope | (setq x 10), (unbind x), {exception thrown unless x exist in a parent scope }|  
| (+ {number}+) | returns the sum of all arguments. | (+ 45) -> 45 (+ 1 3 4 5) -> 13 |
| (- {lvalue} {rvalue}+) | returns the difference of {lvalue} minus the sum of all {rvalue}s. | (- 45) -> -45 (- 10 3) -> 7 |
| (* {number}+) | returns the product of all arguments. | (* 45) -> 45 (* 1 3 4 5) -> 60 |
| (/ {lvalue} {rvalue}+) | returns {lvalue} divided by the product of all {rvalue}s, or F if division by zero occurs. | (/ 10) -> 0.1 (/ 10 10) -> 1 (/ 12 0) -> F |
        """
        map_svalue(dialog_env,"lisp-api", lisp_language_api_description)

        topic_conversation = "top level conversation context about nothing in particular"
        map_svalue(dialog_env, "topic", topic_conversation)
        map_svalue(dialog_env, "user-entry", "The user")
        dialog_seq = """
system: The topic of this conversation is: "{{topic}}".
Your role is of an LLM dialog topic manager specializing on conversing with the user about this topic as well as
taking advantage of a special Lisp runtime environment that is uniquely attached to this conversation.  
{{lisp-api}}
As an LLM, you are able to send {{lisp-name}} code directly to the runtime environment by prefixing your responses with the
meta-role name: "{{lisp-interpreter-role-name}}:".  In this conversation, we will use the concept of meta-roles in order to allow more than just the 
user and the assistant to be participants.  Whenever the first word in your response is a meta-role key followed by a colon, 
the user will forward your response to appropriate meta-role handler.  In the case of the Lisp runtime, there is a meta-role 
handler for the Lisp interpreter called {{lisp-interpreter-role-name}} which will run the remaining message as Lisp code.  
Whenever the first word from the user is a meta-role name, the remaining message body should be interpretted as coming from the
meta-role handler instead of the user.  In this manner, you can send messages directed to any number of participants so long 
as you understand what meta-role key to use to communicate with them.
user: As a test of your understanding, use the {{lisp-name}} runtime to compute 12 plus 234
assistant: {{lisp-interpreter-role-name}}: (+ 12 234)
user: {{lisp-interpreter-role-name}}: {{(+ 12 234)}}
assistant: The {{lisp-name}} has returned {{(+ 12 234)}}
user: Good, now you are not the only LLM topic role manager in the program.  There are others which have their own unique meta-role
names.  For example, your meta-role name is {{my-role}}.  Whenever these other LLM topic managers want to communicate with you, the first
word in their responses must be {{my-role}}: in order for them to send messages to you.  The {{lisp-name}} runtime environment
has a variable called "other-dialog-managers" which stores a list of the meta-role names as well as topics that are being managed
by other LLMs.  Read the value of that variable.
assistant: {{lisp-interpreter-role-name}}: other-dialog-managers
user: {{lisp-interpreter-role-name}}: {{other-dialog-managers}}
assistant: There is one other LLM topic manager specializing on the topic "{{to-do-topic}}" which has a meta-role of {{to-do-role}}.
user: For practice, go ahead and send a message to that other LLM topic manager.  In this case, tell it that I need to finish my 
homework.
assistant: {{to-do-role}}: Finish homework
user: {{to-do-role}}: Okay, I have added "Finish homework" to your to do list
assistant: to do list was updated
user: Similarly to the {{to-do-role}} LLM topic manager, you may also receive a message from an external LLM manager at any given time.
In that case, you should respond to that topic manager using the same meta-role key as was used to communicate with you.  When an external 
LLM dialog manager finishes communicating with you, they will send you a special end-of-dialog key called '{{end-of-message-key}}'.  You should
also send this message when you finish communicating with external topic manager LLMs, however, you don't need to do this with when communicating with 
the Lisp runtime environment using the {{lisp-interpreter-role-name}} meta-role since that system is available at all times.  Go ahead and tell the 
to do list LLM manager bye to close out the conversation with it.  From this point forward, we will be discussing {{topic}}.
assistant: {{to-do-role}}: {{end-of-message-key}}
        """

        resolved_dialog = build_prompt_from_template(dialog_env, dialog_seq)

        print(resolved_dialog)
        print("***** second layer ******")
        resolved_dialog = build_prompt_from_template(dialog_env, resolved_dialog)
        print(resolved_dialog)

        # Test LLM understanding
        from LLMTools import parse_roles_from_dialog_pattern_file, flatten_dialog_list, parse_roles_from_dialog_string

        dialog_pattern = parse_roles_from_dialog_string(resolved_dialog, True)
        print(f"Dialog spec:\n\n{dialog_pattern}")
        query = build_prompt_from_template(dialog_env, "user: {{to-do-role}}: end of dialog with {{my-role}}:")
        response_1 = do_llm_query(dialog_pattern, query)
        print("AI response:\n")
        print(response_1)
        from LLMTools import user_prompt_segment, assistant_prompt_segment
        dialog_pattern.append(user_prompt_segment(query))
        dialog_pattern.append(assistant_prompt_segment(response_1))
        query = "Create a variable called alarms.  Initialize it to an empty list.  We're going to be saving alarms in the dialog's environment"
        response_next = do_llm_query(dialog_pattern, query)
        print("Testing understanding:\n")
        print(response_next)

    def test_can_describe_embedded_lisp_common_prompt(self):
        dialog_env = Environment()
        add_basic_functions(dialog_env)
        add_arithmetic_functions(dialog_env)

        topic = "keeps track of things that the user wants to get done and is doing"
        map_svalue(dialog_env, "to-do-topic", topic)

        map_svalue(dialog_env, "lisp-interpreter-role-name", "slisp")
        map_svalue(dialog_env, "lisp-name", "slisp")
        lisp_language_api_description = """
        This Lisp runtime environment is very simple and only supports a small subset of the functionality of Common Lisp.  Thus, I will refer to this Lisp dialect as
        "{{lisp-name}}" for Simple Lisp.
        ### Values
In {{lisp-name}}, a 'Value' refers to any object instance that can be bound to variables and passed to functions. All Values can be evaluated, transforming them into potentially different Values. {{lisp-name}} categorizes Values into basic and complex types:

- **Basic Types**: Evaluate to themselves and include:
  - **Scalar**: 64-bit integers, 64-bit doubles, strings, booleans (all values are `Truthy` except the singleton `F`) and symbols (unescaped strings).
  - **Composite**: Hashtables with string or integer keys.

- **Complex Types**: May not return verbatim when evaluated:
  - **List**: Delimited by `()`. The first element determines if it's a function/macro call or returns the list itself.
### Functions
{{lisp-name}} only supports a small subset of standard Common Lisp functions as described in the following table:
| Function | Purpose | Example usages |
| --- | --- | --- |
| (setq {variable-name} {value}) | use to assign a value {value} to {variable-name} | (setq x 12) (setq y "twelve") |
| (concat {arg:string}+) | used to concatenate the string arg values into a single string. | (concat "y: " y) (concat "x = " (string x)) |
| (string {arg}) | converts {arg} into a string. | (concat "x = " (string "12")) |
| (list {arg}*) | constructs a list from the arguments passed into the function | (list 1 23 45) |
| (nth {base-list} {index}) | returns the item in position {index} of the list. An exception is thrown if {index} is invalid. | (setq items (list "yes" "no" "maybe")) (nth items 0) -> "yes" |
| (set-nth {base-list} {index} {value}) | destructively updates the value of {base-list} at index {index} to {value}. There must be at least {index} + 1 elements of the list in order to set that value| (setq items (list "yes" "no" "maybe")) (set-nth items 0 "affirmative") -> ("affirmative" "no" "maybe") |
| (length {composite-value}) | returns the number of elements in {composite-value}. | (length (make-int-hashtable (list (1 "one") (2 "two") (3 "three")))) -> 3 (length (list 1 3 4 5 5)) -> 5 |
| (progn {body-exp}+) | executes each {body-exp} sequentially and returns the value of the last expression. | (progn (setq x 10) (setq y 20)) -> 20 (if (= 12 12) (progn (setq z 1) (setq x (+ x z)))) -> 11 |
| (append-item {arg:list} {item}) | appends {item} to the end of {arg} | (setq foods ()) (setq foods (append-item foods "milk")) (setq foods (append-item foods "water")) |
| (unbind {var-name:symbol}) | Unbinds the {var-name} in the outermost environment within scope | (setq x 10), (unbind x), {exception thrown unless x exist in a parent scope }|  
| (+ {number}+) | returns the sum of all arguments. | (+ 45) -> 45 (+ 1 3 4 5) -> 13 |
| (- {lvalue} {rvalue}+) | returns the difference of {lvalue} minus the sum of all {rvalue}s. | (- 45) -> -45 (- 10 3) -> 7 |
| (* {number}+) | returns the product of all arguments. | (* 45) -> 45 (* 1 3 4 5) -> 60 |
| (/ {lvalue} {rvalue}+) | returns {lvalue} divided by the product of all {rvalue}s, or F if division by zero occurs. | (/ 10) -> 0.1 (/ 10 10) -> 1 (/ 12 0) -> F |
        """
        map_svalue(dialog_env, "lisp-api", lisp_language_api_description)

        topic_conversation = "top level conversation context about nothing in particular"
        map_svalue(dialog_env, "topic", topic_conversation)
        dialog_seq = """
system: The topic of this conversation is: "{{topic}}".
Your role is of an LLM dialog topic manager specializing on conversing with the user about this topic as well as
taking advantage of a special Lisp runtime environment that is uniquely attached to this conversation.  
{{lisp-api}}
As an LLM, you are able to send {{lisp-name}} code directly to the runtime environment by prefixing your responses with the
meta-role name: "{{lisp-interpreter-role-name}}:".  In this conversation, we will use the concept of meta-roles in order to allow more than just the 
user and the assistant to be participants.  Whenever the first word in your response is {{lisp-interpreter-role-name}}:, 
the user will forward your response to the Lisp runtime.  The runtime will process the lisp code in the remaining message and return the
response to the user, who will forward that response to you by setting the first word in their response to {{lisp-interpreter-role-name}}:
user: As a test of your understanding, use the {{lisp-name}} runtime to compute 12 plus 234
assistant: {{lisp-interpreter-role-name}}: (+ 12 234)
user: {{lisp-interpreter-role-name}}: {{(+ 12 234)}}
assistant: The {{lisp-name}} has returned {{(+ 12 234)}}
        """

        resolved_dialog = build_prompt_from_template(dialog_env, dialog_seq)

        print(resolved_dialog)
        print("***** second layer ******")
        resolved_dialog = build_prompt_from_template(dialog_env, resolved_dialog)
        print(resolved_dialog)

        # Test LLM understanding
        from LLMTools import parse_roles_from_dialog_pattern_file, flatten_dialog_list, \
            parse_roles_from_dialog_string

        dialog_pattern = parse_roles_from_dialog_string(resolved_dialog, True)
        print(f"Dialog spec:\n\n{dialog_pattern}")
        query = "Create a variable called to-dos.  Initialize it to an empty list.  We're going to be saving a to-do list in the dialog's environment"
        response_1 = do_llm_query(dialog_pattern, query)
        print("AI response to create to-do variable:\n")
        print(response_1)
        forwarded_lisp = get_forwarded_message(response_1)
        if forwarded_lisp:
            from LLMTools import user_prompt_segment, assistant_prompt_segment
            dialog_pattern.append(user_prompt_segment(query))
            dialog_pattern.append(assistant_prompt_segment(response_1))
            auto_query = "{{lisp-interpreter-role-name}}: {{" + forwarded_lisp + "}}"
            try:
                auto_query = build_prompt_from_template(dialog_env, auto_query)
            except Exception as e:
                exception_message = f"{e}"
                auto_query = "{{lisp-interpreter-role-name}}: {{\"" + exception_message + "\"}}"
                auto_query = build_prompt_from_template(dialog_env, auto_query)
            response_next = do_llm_query(dialog_pattern, auto_query)
            print("Response to create to-do list")
            print(response_next)
            dialog_pattern.append(user_prompt_segment(auto_query))
            dialog_pattern.append(assistant_prompt_segment(response_next))

            query = "Now add a new item to my to do list called brush my teeth."

            response_next = do_llm_query(dialog_pattern, query)
            dialog_pattern.append(user_prompt_segment(query))
            dialog_pattern.append(assistant_prompt_segment(response_next))
            forwarded_lisp = get_forwarded_message(response_next)
            if forwarded_lisp:
                print("Testing understanding of mechanism for tracking to do items:\n")
                print(response_next)
                auto_query = "{{lisp-interpreter-role-name}}: {{" + forwarded_lisp + "}}"
                try:
                    auto_query = build_prompt_from_template(dialog_env, auto_query)
                except Exception as e:
                    exception_message = f"{e}"
                    auto_query = "{{lisp-interpreter-role-name}}: {{\"" + exception_message + "\"}}"
                    auto_query = build_prompt_from_template(dialog_env, auto_query)
                response_next = do_llm_query(dialog_pattern, auto_query)
                dialog_pattern.append(user_prompt_segment(auto_query))
                dialog_pattern.append(assistant_prompt_segment(response_next))
                print("Response from adding an item to the list")
                print(response_next)

                query = "Let's check things by reading the items in our to-do list so far"
                response_next = do_llm_query(dialog_pattern, query)
                print("Response when checking to-dos\n")
                print(response_next)
                dialog_pattern.append(user_prompt_segment(query))
                dialog_pattern.append(assistant_prompt_segment(response_next))
                forwarded_lisp = get_forwarded_message(response_next)
                if forwarded_lisp:
                    auto_query = "{{lisp-interpreter-role-name}}: {{" + forwarded_lisp+ "}}"
                    try:
                        auto_query = build_prompt_from_template(dialog_env, auto_query)
                    except Exception as e:
                        exception_message = f"{e}"
                        auto_query = "{{lisp-interpreter-role-name}}: {{\"" + exception_message + "\"}}"
                        auto_query = build_prompt_from_template(dialog_env, auto_query)
                        print("Final command was a failure!!!")
                    response_next = do_llm_query(dialog_pattern, auto_query)
                    print("Response final\n")
                    print(response_next)
                    dialog_pattern.append(user_prompt_segment(auto_query))
                    dialog_pattern.append(assistant_prompt_segment(response_next))
                else:
                    print("AI failure to produce lisp!!")
            else:
                print("AI failed to produce command to add item to-do list")
        else:
            print("AI failed to produce command to create to-do list")



def get_forwarded_message(message:str, target_prefix:str = "slisp:"):
    if message.strip().startswith(target_prefix):
        return message[len(target_prefix):]
    else:
        return None



def do_llm_query(prior, query):
    from LLMTools import system_prompt_segment


    # OpenAI
    model_name, url, key = CONFIG_MAP["model-config"]["openai"]


    # Models working with ollama:
    # | Model | Score | platform | comment |
    # | --- | --- | --- | --- |
    # | gemma3:4b | 5 | ollama |Fully works for this example with 128K context and 3.3GB |
    # | gpt-4o | 5+ | openai | Fully trustable with 128K context but price is high |
    # | llama3:latest | 3 | ollama | Only works with auto-prompting.  May improve performance in subsequent rounds |
    # | meta-llama-3-8b-instruct | 4.7 | lm-studio| Didn't fully understand lisp api put code was formatted correctly |
    #
    # partially working models: (gets right answer with auto-correction)
    # llama3:latest
    # Not so great but possibly useful (get's right answer but wrong formatting without prompting)
    # gemma3:4b-it-qat

    # Ollama models
    model_name = "gemma3:4b"
    url = "http://localhost:11434/v1/"
    key = "ollama"

    # LM Studio models
    #model_name = "phi-4-mini-instruct"
    #url = "http://127.0.0.1:1234/v1/"
    #key = "ollama"

    if len(prior) == 0:
        i = [system_prompt_segment("You are an AI agent whose purpose is to be as concise, helpful and accurate as possible")] + prior
    else:
        i = prior
    return do_multi_shot_llm_query(i, query, llm_name=model_name, url = url, api_key=key)

def map_svalue(env, key, value):
    env.map_value(key, to_svalue(value))

def map_nvalue(env, key, value):
    env.map_value(key, LispTools.make_integer(value))

if __name__ == '__main__':
    unittest.main()
