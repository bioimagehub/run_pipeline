import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hello", nargs='+', type=str, help="")
    
    parsed_args: argparse.Namespace = parser.parse_args()

    print(parsed_args.hello)