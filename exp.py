from exp_objects import *
import argparse
parser = argparse.ArgumentParser()
def main(input):
    data0 = pd.read_csv("./saves/GP0.csv", index_col=0)
    print('data\n%s'%(data0.tail(5)))

parser.add_argument("input")
args = parser.parse_args()

main(args.input)
