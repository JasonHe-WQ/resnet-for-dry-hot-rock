import pandas as pd
import matplotlib.pyplot as plt

step = 200


# Reload the CSV file due to execution environment reset
def plotOneGain(gain, length):
    fileName1 = f'{gain}_modify'
    fileName2 = f'{gain}'
    print("plotting: ", fileName1, " and ", fileName2, "with length", length)
    plotOneDataset(fileName1, length)
    plotOneDataset(fileName2, length)


def plotOneDataset(fileName, length):
    filePath = f'{fileName}/{fileName}.csv'
    data = pd.read_csv(filePath, header=None)  # Assuming no header as per user's description
    nrows = data.shape[1] // (length*2)

    # Plotting the specified ranges in separate plots
    fig, axes = plt.subplots(nrows, 1, figsize=(15, 60))

    for i in range(nrows):
        start_index = 50 + i * step
        end_index = start_index + length - 1
        axes[i].plot(data.iloc[0, start_index:end_index + 1])  # +1 to include end_index
        axes[i].set_title(f'Line Plot of Data from Index {start_index} to {end_index}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    plt.tight_layout()
    plt.savefig(f'./{fileName}/{fileName}.png')
    # plt.show()

    # Extract the first row for plotting
    first_row = data.iloc[-1]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(first_row)
    plt.title('Line Plot of the First Row of Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'./{fileName}/{fileName}_line.png')
    # plt.show()


if __name__ == '__main__':
    plotOneGain('0dB', 100)
    plotOneGain('-5dB', 100)
    plotOneGain('-10dB', 100)
    plotOneGain('-15dB', 100)
