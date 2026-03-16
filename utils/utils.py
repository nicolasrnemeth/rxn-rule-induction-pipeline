# --------------------------------------------------------------------
# Original code Copyright (c) 2025 Nicolas Nemeth licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

import re
from typing import List, Dict, Set, Tuple


def replace_numbers_in_string(string: str, number_map: Dict[int, int], replace_pattern: str="(?<=:)\\d+(?=\\])") -> str:
    """
    Replaces numbers found within string based on a mapping dictionary
    where keys and values are integers.

    Args:
        string (str): A string to process.
        number_map (Dict[int, int]): A dictionary where k:v (integer:integer).
        replace_pattern: A regular expression string defining the specific
            number patterns to target for replacement.
            Regex to only match SMILES atom maps, but not numbers related to ring closure indication: `"(?<=:)\\d+(?=\\])"`.
            Defaults to `"\\d+"` (replace all numbers).

    Returns:
        A new string with numbers replaced according to the map.
        Numbers found in the string but not present as keys in the map
        will remain unchanged.
    """
    return replace_numbers_in_list([string], number_map, replace_pattern)[0]
    

def replace_numbers_in_list(string_list: List[str], number_map: Dict[int, int], replace_pattern: str="\\d+") -> List[str]:
    """
    Replaces numbers found within strings in a list based on a mapping dictionary
    where keys and values are integers.

    Args:
        string_list: A list of strings to process.
        number_map: A dictionary where k:v (integer:integer).
        replace_pattern: A regular expression string defining the specific
            number patterns to target for replacement.
            Regex to only match SMILES atom maps, but not numbers related to ring closure indication: `"(?<=:)\\d+(?=\\])"`.
            Defaults to `"\\d+"` (replace all numbers).

    Returns:
        A new list of strings with numbers replaced according to the map.
        Numbers found in the strings but not present as keys in the map
        will remain unchanged.
    """
    processed_list = []
    replace_regex = re.compile(replace_pattern)
    any_number_regex = re.compile(r'\d+')

    def replace_match(match):
        number_str = match.group(0)
        try:
            number_int = int(number_str)
            replacement_value = number_map.get(number_int)
            if replacement_value is None:
                return number_str
            return str(replacement_value)
        except ValueError:
            print(f"Warning: Could not convert '{number_str}' to int. Skipping.")
            return number_str

    for s in string_list:
        # lines with no numbers need no mapping
        if not any_number_regex.search(s):
            processed_list.append(s)
            continue
        processed_string = replace_regex.sub(replace_match, s)
        processed_list.append(processed_string)

    return processed_list