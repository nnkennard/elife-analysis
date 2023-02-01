import json
import sys

def main():

    path_file, output_dir = sys.argv[1:]

    with open(path_file, 'r') as f:
        file_list = json.load(f)
        for set_name, paths in file_list.items():
            fmt = paths['meta']['format']
            subset = paths['meta']['subset']
            for annotator, input_path in paths.items():
                if  annotator == "meta":
                    continue
                print(annotator, set_name, input_path)
                with open(f'{output_dir}/{subset}/{set_name}_{annotator}_{fmt}.csv', 'w') as g:
                    with open(input_path, 'r') as h:
                        g.write(h.read())


if __name__ == "__main__":
    main()
