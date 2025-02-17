import re
from tqdm import tqdm
import numpy as np
from scipy.signal import savgol_filter


def extract_numbers(line):
    """Extract all numbers (ints and floats) from string."""
    pattern = re.compile(r"[-+]?(?:\d*\.*\d+)")
    matches = pattern.findall(line)
    return matches

def extract_keywords(line):
    """Extract all words containing underscore in string."""
    pattern = re.compile(r'(\b\w+_?\w*_?\w*_?\w*)=')
    keywords = pattern.findall(line)
    return keywords


def read_data(*file_paths):
    """Read data from log file into dictionary. Padding with NaNs
    if values are missing.
    """
    content = ''
    for file_path in file_paths:
        # Read the content of the text file
        with open(file_path, 'r') as file:
            content += file.read()

    lines = content.split('\n')

    # create dictionary with all static keywords, expected in all lines
    data = {'epoch': [], 'iteration': [], 'elapsed_time': [], 'estimated_time': [], 'speed': []}
    nline = 0
    for line in tqdm(lines):
        if not line.startswith('Epoch'):
            continue
        numbers = extract_numbers(line) # extract all numbers from line
        # continue if only static numbers
        if len(numbers) < 7:
            continue
        data['epoch'].append(float(numbers[0]))
        data['iteration'].append(float(numbers[1]))
        # hours if numbers > 11, else just seconds and minutes
        if len(numbers) > 11:
            data['elapsed_time'].append(60*float(numbers[2])+float(numbers[3]))
            data['estimated_time'].append(3600*float(numbers[4])+60*float(numbers[5])+float(numbers[6]))
            next_ = 7
        else:
            data['elapsed_time'].append(60*float(numbers[2])+float(numbers[3]))
            data['estimated_time'].append(60*float(numbers[4])+float(numbers[5]))
            next_ = 6
        data['speed'].append(float(numbers[next_]))
        # get all numbers and keywords that are not static
        dynamic_keywords = extract_keywords(line)
        dynamic_numbers = list(map(float, numbers[next_+1:]))
        for keyword, number in zip(dynamic_keywords, dynamic_numbers):
            try:
                data[keyword].append(number)
            except KeyError:
                data[keyword] = [number]
        nline += 1

    # pad to standard lengths
    for key in data.keys():
        key_len = len(data[key])
        diff = nline - key_len 
        if diff > 0:
            data[key] = diff * [np.nan] + data[key]
        data[key] = np.asarray(data[key], dtype=np.float32)
    return data
                                                    
def get_total_iteration(data):
    """Return total iterations."""
    epochs = data['epoch']
    iterations = data['iteration']
    max_iterations = iterations.max()
    total_iteration = max_iterations * epochs + iterations
    return total_iteration

if __name__ == "__main__":
    val_loss = []
    total_iterations = []
    for filename in [
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_2.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_3.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_4.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_4mem.out',
            '/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d.out',
            '/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d_2.out',
            '/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d_3.out',
            '/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d_4.out',
        ]:
        data = read_data(filename)

        total_iteration = get_total_iteration(data)
        total_iterations.append(total_iteration)
        val_loss.append(data['train_mse_step'])

    import matplotlib.pyplot as plt

    labels = ["roll4, lr=5e-7", "roll2, lr=5e-7", "roll2, lr=2.5e-7", "roll4, lr=2.5e-7"]

    for total_iteration, loss, label in zip(total_iterations, val_loss, labels):
        loss = savgol_filter(loss, 5, 2)
        plt.plot(total_iteration, loss, label=label)
    plt.legend(loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("CRPS")
    plt.title("Ensemble rollout")
    plt.show()
