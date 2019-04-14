import pprint

def format_arms(arms):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pformat(arms)

def print_arms(arms):
    print(format_arms(arms))
