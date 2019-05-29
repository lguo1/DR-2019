from exp_objects import *
import argparse
parser = argparse.ArgumentParser()
def main(input):
    test()

parser.add_argument("input")
args = parser.parse_args()

main(args.input)
