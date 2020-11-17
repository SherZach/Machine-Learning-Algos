#!/usr/bin/python3
import sys

# Outputs TSVs, where the word is the key and the total number of occurances of that word is the value
# Example input: a  1
#                a  1
#                a  1
#                ....
#                core   1
#                core   1
#                etc.
# Example output: a 3
#                 core  2

key = ""
count = 0
reduced = ""

for input_ in sys.stdin:
    mapped = input_.split(sep="\t")
    word = mapped[0]
    num = int(mapped[1][0])
    if key != word:
        if key:
            reduced += key + "\t" + str(count) + "\n"
        key = word
        count = num
    elif key == word:
        count += num
reduced += key + "\t" + str(count) + "\n" # Because it doesn't print the last word as the loop quits when input_ == none
sys.stdout.write(reduced)