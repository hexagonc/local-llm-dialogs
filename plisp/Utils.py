
import os
import datetime

def read_prompt_file(prompt_file_name):
    with open(prompt_file_name, 'r') as file:
        return file.read()


default_text_file_ext_list = ['.txt', '.md', '.rst', '.html', '.htm', '.xml', '.json', '.csv', '.tsv', '.yaml', '.yml',
                              '.log', '.ini', '.cfg', '.conf', '.properties', '.java', '.js', '.ts', '.py', '.sh',
                              '.bat', '.cmd', '.ps1', '.psm1', '.psd1', '.ps1xml', '.pssc', '.pssc', '.pss', '.gradle']
default_text_file_ext_set = set(default_text_file_ext_list)


def read_text(f):
    with open(f, 'r') as file:
        return file.read()


def file_path_to_list(file_path, content_included_path_names_set=None, excluded_folder_names_set=None,
                      included_file_extension_set=None, text_file_extension_set=None):
    if excluded_folder_names_set is None:
        excluded_folder_names_set = set()

    if text_file_extension_set is None:
        text_file_extension_set = default_text_file_ext_set

    name = os.path.basename(file_path)
    create_date_timestamp = format_timestamp(os.path.getctime(file_path))
    modify_date_timestamp = format_timestamp(os.path.getmtime(file_path))

    if os.path.isfile(file_path):
        include_contents = content_included_path_names_set is not None and file_path in content_included_path_names_set
        if not include_contents:
            include_contents = True
            ext = os.path.splitext(file_path)[1]
            if included_file_extension_set is not None:
                if ext not in included_file_extension_set:
                    include_contents = False
        if include_contents:
            contents = read_text(file_path)
        else:
            contents = None
        # Contents = None means that the file contents was not read
        return [name, create_date_timestamp, modify_date_timestamp, contents]
    include_contents = content_included_path_names_set and file_path in content_included_path_names_set or not content_included_path_names_set and name not in excluded_folder_names_set

    if not include_contents:
        contents = None
    else:
        contents = [file_path_to_list(file_path + "\\" + f, content_included_path_names_set, excluded_folder_names_set,
                                      included_file_extension_set, text_file_extension_set) for f in
                    os.listdir(file_path)]
    return [name, create_date_timestamp, modify_date_timestamp, contents]


def delimit_text_content(s):
    delimiter = "\""
    while s.find(delimiter) >= 0:
        delimiter = delimiter + "\""
    return f"{delimiter}{s}{delimiter}"


def delimit_text_content_for_lisp(s):
    if s is None:
        return "F"
    return delimit_text_content(s)


# format as YYYY-MM-DD HH:MM:SS
def format_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def file_path_to_lisp(file_path, content_included_path_names_set=None, excluded_folder_names_set=None,
                      included_file_extension_set=None, text_file_extension_set=None):
    if excluded_folder_names_set is None:
        excluded_folder_names_set = set()

    if text_file_extension_set is None:
        text_file_extension_set = default_text_file_ext_set

    name = os.path.basename(file_path)
    create_date_timestamp = format_timestamp(os.path.getctime(file_path))
    modify_date_timestamp = format_timestamp(os.path.getmtime(file_path))
    if name in excluded_folder_names_set:
        return None

    if os.path.isfile(file_path):
        ext = os.path.splitext(file_path)[1]
        if included_file_extension_set is not None:
            if ext not in included_file_extension_set:
                return None
        include_contents = content_included_path_names_set is not None and file_path in content_included_path_names_set
        if not include_contents:
            include_contents = True
            ext = os.path.splitext(file_path)[1]
            if text_file_extension_set is not None:
                if ext not in text_file_extension_set:
                    include_contents = False
        if include_contents:
            contents = read_text(file_path)
        else:
            contents = None
        # Contents = None means that the file contents was not read
        return f"(\"{name}\", {create_date_timestamp}, {modify_date_timestamp}, {delimit_text_content_for_lisp(contents)})"
    include_contents = content_included_path_names_set and file_path in content_included_path_names_set or not content_included_path_names_set and name not in excluded_folder_names_set

    if not include_contents:
        contents = None
        return f"(\"{name}\" {create_date_timestamp} {modify_date_timestamp} F)"
    else:
        contents = [file_path_to_lisp(file_path + "\\" + f, content_included_path_names_set, excluded_folder_names_set,
                                      included_file_extension_set, text_file_extension_set) for f in
                    os.listdir(file_path)]
    contents = [c for c in contents if c is not None]
    list_p = " ".join(contents)
    return f"(\"{name}\" {create_date_timestamp} {modify_date_timestamp} ({list_p}))"


def tree(dir_path, prefix='', excluded_folder_names_set=None):
    if not os.path.isdir(dir_path):
        return "Invalid directory path"

    # Set default for the optional parameter
    if excluded_folder_names_set is None:
        excluded_folder_names_set = set()

    # Collect the lines to print
    tree_lines = []
    items = os.listdir(dir_path)
    items = sorted(items, key=lambda s: s.lower())  # Sort items alphabetically

    for index, item in enumerate(items):
        item_path = os.path.join(dir_path, item)

        # Exclude folders
        if os.path.isdir(item_path) and item in excluded_folder_names_set:
            continue

        connector = '├── ' if index < len(items) - 1 else '└── '
        tree_lines.append(f"{prefix}{connector}{item}")

        if os.path.isdir(item_path):
            extension = '│   ' if index < len(items) - 1 else '    '
            tree_lines.append(tree(item_path, prefix + extension, excluded_folder_names_set))

    return "\n".join(tree_lines)

def list_all_file_names(path=None):
    if path is None:
        path = os.getcwd()
    return os.listdir(path)
