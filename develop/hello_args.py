import argparse

def hello(positional_arg, flag_arg):
    print(f"Hello, {positional_arg}!")
    if flag_arg:
        print(f"You've provided the flag arg: {flag_arg}")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Hello Argument Script")
    parser.add_argument("name", type=str, help="The name to greet")  # Positional argument
    parser.add_argument("-g", "--greeting", type=str, help="Optional greeting flag argument")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the function with parsed arguments
    hello(args.name, args.greeting)

if __name__ == "__main__":
    main()
