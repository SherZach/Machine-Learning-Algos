#!/usr/bin/python3
import sys


# Reads in strings from stdin
#
# Outputs every word to stdout in a TSV format
# where 1 is the value.
#
# Example input: "Lorem Ipsum...? I don't know the rest of it"
# Example output: lorem 1
#                 ipsum...? 1
#                 i 1
#                 etc.

for input_ in sys.stdin:
    text = input_.lower().split()
    output = ""
    for word in text:
        output += word + "\t" + "1" + "\n"
    sys.stdout.write(output)
