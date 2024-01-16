import pandas as pd
import numpy as np
import preprocess
import plt




def create_and_save_dataframe(m, n, file_name, fill_value=True):
    # 创建一个所有值为 fill_value 的 DataFrame
    df = pd.DataFrame(fill_value, index=range(m), columns=range(n))
    print("labels shape", df.shape)
    # 将DataFrame保存为CSV文件
    df.to_csv(file_name, index=False,header=False)

    return df

def modify_data(gain, length):
    fileName = f'{gain}_modify'
    filePath = f'./{gain}/{gain}.csv'
    data = pd.read_csv(filePath, header=None)
    dataShape = data.shape
    dataCol = data.shape[1]
    print("total columns", dataCol, "length for each segment", length)
    segmentsNum = dataCol // length
    dataRow = data.shape[0]
    create_and_save_dataframe(dataRow, segmentsNum-1, f'./{gain}/{gain}-labels.csv',1)
    create_and_save_dataframe(dataRow, segmentsNum-1, f'./{fileName}/{fileName}-labels.csv',0)

    def replace_with_noise(data_segment, threshold=3):
        data_segment = list(data_segment)
        length = len(data_segment)
        # Length of the segment to be analyzed and possibly replaced
        analyze_length = int(0.8 * length)

        # Indices for the middle segment
        end_analyze_index = length - analyze_length // 2
        start_analyze_index = analyze_length // 2

        # Calculate mean and std of the middle segment
        middle_segment = data_segment[start_analyze_index:end_analyze_index]
        mean, std = np.mean(middle_segment), np.std(middle_segment)

        # Function to generate noise
        def generate_constrained_noise(size, mean, std, threshold):
            noise = np.random.normal(mean, std, size)
            noise = np.clip(noise, mean - threshold * std, mean + threshold * std)
            return noise

        # Replace from end_analyze_index to the end of the segment if deviation is more than threshold stds
        for i in range(end_analyze_index, length):
            if abs(data_segment[i] - mean) > threshold * std:
                noise_size = length - i
                noise = generate_constrained_noise(noise_size, mean, std, threshold)
                data_segment[i:length] = noise
                break

        # Replace from the start to start_analyze_index if deviation is more than threshold stds
        for i in range(start_analyze_index, -1, -1):
            if abs(data_segment[i] - mean) > threshold * std:
                noise_size = i + 1
                noise = generate_constrained_noise(noise_size, mean, std, threshold)
                data_segment[0:i + 1] = noise
                break

        return data_segment

    modified_rows = []
    modified_segments = []
    for j in range(len(data)):
        # Processing each segment as specified
        for i in range(segmentsNum):
            start_index = i * length
            end_index = start_index + length - 1
            segment = data.iloc[j, start_index:end_index + 1].copy()
            modified_segment = replace_with_noise(segment)
            modified_segments.append(modified_segment)

        # Combining the modified segments into a single row
        modified_segments = [pd.Series(s) for s in modified_segments]
        combined_modified_data_1600 = pd.concat(modified_segments)
        modified_rows.append(combined_modified_data_1600)
        combined_modified_data_1600 = None
        modified_segments = []

    # Writing the combined data to a CSV file
    modified_rows = [s.reset_index(drop=True) for s in modified_rows]
    df = pd.concat(modified_rows, axis=1).T
    print("modified data shape", df.shape)
    assert df.shape == dataShape
    df.to_csv(f'./{fileName}/{fileName}.csv', index=False, header=False)


    plt.plotOneGain(gain, length)


if __name__ == '__main__':
    preprocess.preProcess('0dB')
    modify_data('0dB', 100)
    preprocess.preProcess('-5dB')
    modify_data('-5dB', 100)
    preprocess.preProcess('-10dB')
    modify_data('-10dB', 100)
    preprocess.preProcess('-15dB')
    modify_data('-15dB', 100)
