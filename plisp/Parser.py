from Value import Value, ValueType, IntValue, ListValue, StringValue, FloatValue, NullValue
from FunctionTemplate import FunctionTemplate

NULL_LITERAL = "F"
NULL_VALUE = NullValue()


def parse(exp, error_on_incomplete_parser=False):
    resultList = []
    index = 0
    last_index = 0
    result = prefix_parse(exp, index)
    while result is not None and result[0] is not None:
        resultList.append(result[0])
        last_index = index
        index = result[1]
        if index < len(exp):
            result = prefix_parse(exp, index)
        else:
            break

    if result is None and error_on_incomplete_parser:
        raise Exception(f"Parser failed at index {last_index} with input {exp}")
    if len(resultList) == 0:
        return None
    return resultList



def prefix_parse(input, start):
    # leading whitespace
    while start < len(input) and input[start].isspace():
        start += 1

    # can't match for tokens at end of string
    if start == len(input):
        return None, start

    # check for quoted
    if input[start] == '\'':
        rest = prefix_parse(input, start + 1)
        if rest is not None and rest[0] is not None:
            return (ListValue([StringValue("quote", is_symbol=True), rest[0]]), rest[1])

    # check for back-quoted
    if input[start] == '`':
        rest = prefix_parse(input, start + 1)
        if rest is not None and rest[0] is not None:
            return (ListValue([StringValue("back-quote", is_symbol=True), rest[0]]), rest[1])

    # check for comma delimited
    if input[start] == ',' and start < len(input) - 1:
        rest = prefix_parse(input, start + 1)
        if rest is not None and rest[0] is not None:
            rest[0].set_comma_delimited()
            return rest

    # try numeric
    index = start
    num = ""
    contains_decimal = False
    contains_integer = False
    contains_fraction = False
    contains_negative = False
    contains_exponent = False
    while index < len(input):
        c = input[index]
        if c.isdigit():
            if contains_decimal:
                contains_fraction = True
            if not contains_fraction:
                contains_integer = True
            num += c
        elif c == '-':
            if not contains_negative and \
                    (contains_exponent and not contains_integer or not contains_integer and not contains_decimal and not contains_fraction):
                contains_negative = True
            else:
                break
            num += c
        elif c == '.':
            if not contains_decimal:
                contains_decimal = True
                num += c
            else:
                break
        elif c == 'e' or c == 'E':
            if not contains_exponent and contains_integer:
                contains_exponent = True
                contains_negative = False
                contains_integer = False
                num += c
            else:
                break
        elif c.isspace() or c == '(' or c == ')':
            if contains_fraction or contains_integer and not contains_decimal:
                if contains_fraction:
                    return (FloatValue(float(num)), index)
                else:
                    return (IntValue(int(num)), index)
            else:
                break
        else:
            break
        if index == len(input) - 1:
            if contains_fraction or contains_integer and not contains_decimal:
                if contains_fraction:
                    return (FloatValue(float(num)), index + 1)
                else:
                    return (IntValue(int(num)), index + 1)
        index += 1

    # s-expression
    index = start
    listArgs = []
    result_value = None
    result_next_start = None
    if input[index] == '(':
        index += 1
        while index < len(input):
            if input[index].isspace():
                index += 1
                continue
            else:
                if input[index] == ')':
                    return (ListValue(listArgs), index + 1)
            result = prefix_parse(input, index)
            if result is not None and result[0] is not None:
                listArgs.append(result[0])
                index = result[1]
            else:
                return None
        return None

    # String
    index = start
    previous_delimiter = False
    string = ""
    if input[index] == '"':
        index += 1
        while index < len(input):
            c = input[index]
            if c == '"':
                if previous_delimiter:
                    string += c
                    previous_delimiter = False
                else:
                    return (StringValue(string), index + 1)
            elif c == '\\':
                if previous_delimiter:
                    string += c
                previous_delimiter = not previous_delimiter
            else:
                if previous_delimiter:
                    previous_delimiter = False
                string += c
            index += 1

    # Identifiers
    index = start
    id = ""
    c = input[index]
    if not c.isdigit() and c != ')':
        id += c
        index += 1
        while index < len(input):
            c = input[index]
            if not c.isspace() and c != ')' and c != '(' and c != '"':
                id += c
            else:
                return (StringValue(id, is_symbol=True), index)
            index += 1
        return (StringValue(id, is_symbol=True), index)
    return None