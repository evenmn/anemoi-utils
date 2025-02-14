import re
from tqdm import tqdm
import numpy as np


def extract_numbers(line):
    pattern = re.compile(r"[-+]?(?:\d*\.*\d+)")
    matches = pattern.findall(line)
    return matches

def extract_keywords(line):
    pattern = re.compile(r'(\b\w+_?\w*_?\w*_?\w*)=')
    keywords = pattern.findall(line)
    return keywords


def make_table(*file_paths):
    content = ''
    for file_path in file_paths:
        # Read the content of the text file
        with open(file_path, 'r') as file:
            content += file.read()

    lines = content.split('\n')

    numbers_float = []
    all_keywords = ["epoch", "iteration", "elapsed_time", "estimated_time", "speed"]
    for line in tqdm(lines):
        if not line.startswith('Epoch'):
            continue
        lst = []
        numbers = extract_numbers(line)
        if len(numbers) < 7:
            continue
        lst.append(float(numbers[0]))
        lst.append(float(numbers[1]))
        if len(numbers) > 11:
            lst.append(60*float(numbers[2])+float(numbers[3]))
            lst.append(3600*float(numbers[4])+60*float(numbers[5])+float(numbers[6]))
            next_ = 7
        else:
            lst.append(60*float(numbers[2])+float(numbers[3]))
            lst.append(60*float(numbers[4])+float(numbers[5]))
            next_ = 6
        lst.extend(map(float, numbers[next_:]))
        numbers_float.append(lst)
        all_keywords.extend(extract_keywords(line))
        all_keywords = list(dict.fromkeys(all_keywords))

    # pad
    key_len = len(all_keywords)
    for it in range(len(numbers_float)):
        this_len = len(numbers_float[it])
        if this_len < key_len:
            diff = key_len - this_len
            numbers_float[it].extend(diff * [np.nan])

    for it in reversed(range(len(numbers_float))):
        if len(numbers_float[it]) != key_len:
            del numbers_float[it]

    return np.asarray(numbers_float, dtype=np.float32), all_keywords

def key_to_arr(key, table, keywords):
    assert key in keywords

    idx = keywords.index(key)
    return table[:,idx]
                                                    

if __name__ == "__main__":
    epochs = []
    val_loss = []
    for filename in [
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_2.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_3.out',
            #'/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_c_safcrps_k5_s1_4.out',
            '/pfs/lustrep3/scratch/project_465000454/nordhage/ens-score-anemoi/logs/ni3_d.out',
        ]:
        table, keywords = make_table(filename)
        np.set_printoptions(threshold=np.inf)
        epochs.append(key_to_arr('epoch', table, keywords))
        val_loss.append(key_to_arr('train_mse_step', table, keywords))


    epochs = np.concatenate(epochs, axis=0)
    val_loss = np.concatenate(val_loss, axis=0)

    print(val_loss)

    #epochs = epochs[np.isfinite(val_loss)]
    #val_loss = val_loss[np.isfinite(val_loss)]

    xy = (epochs, val_loss)
    #np.save('/leonardo/home/userexternal/enordhag/loss_ni1_a_4m_bs32.npy', xy)

    import matplotlib.pyplot as plt

    #plt.plot(*xy)
    plt.plot(val_loss)
    plt.xlabel("Steps")
    plt.ylabel("Spatial almost fair CRPS")
    plt.title("Stage C")
    plt.show()
