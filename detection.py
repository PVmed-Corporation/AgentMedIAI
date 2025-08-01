import carinanet
import argparse

def detection(input,output):
    model = carinanet.CarinaNetModel()
    result = model.predict(input)
    print(f"Carina: {result['carina']}")
    print(f"ETT: {result['ett']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path')
    parser.add_argument('-output_path')
    args = parser.parse_args()

    detection(args.input_path,args.output_path)