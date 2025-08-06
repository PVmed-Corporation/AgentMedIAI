import carinanet
import argparse

def detection(input):
    model = carinanet.CarinaNetModel()
    result = model.predict(input)
    print(f"Carina: {result['carina']}")
    print(f"ETT: {result['ett']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path')
    args = parser.parse_args()

    detection(args.input_path)