import numpy as np
from tqdm import tqdm


def preprocess_path(path_path: str, output_path: str):
    cleaned_inputs = []
    cleaned_outputs = []

    npzfile = np.load(path_path)
    for key in tqdm(npzfile.files, leave=False):
        path = npzfile[key]
        goal = path[-1]
        N = path.shape[0]

        cleaned_input = np.zeros((N-1, 6), dtype=np.float32)
        cleaned_input[:, :3] = path[:-1]
        cleaned_input[:, 3:] = goal
        cleaned_inputs.append(cleaned_input)

        cleaned_outputs.append(path[1:].astype(np.float32))

    cleaned_inputs = np.concatenate(cleaned_inputs)
    cleaned_outputs = np.concatenate(cleaned_outputs)
    np.savez(output_path, input=cleaned_inputs, output=cleaned_outputs)


def main():
    for i in tqdm(range(40)):
        preprocess_path("./data/train/env{}_path.npz".format(i),
                        "./data/train/env{}_cleaned.npz".format(i))


if __name__ == "__main__":
    main()
