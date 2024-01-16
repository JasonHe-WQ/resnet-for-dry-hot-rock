import pandas as pd

# 读取csv文件


def preProcess(gain):
    print(f"==={gain} Processing===")
    data = pd.read_csv(f'{gain}/{gain}.csv', header=None)
    lastLine = data.iloc[-1]
    if lastLine.isna().any():
        data.iloc[-1] = data.iloc[-2]
    data.to_csv(f'{gain}/{gain}.csv', index=False, header=False)



if __name__ == '__main__':
    preProcess('0dB')
    preProcess('-5dB')
    preProcess('-10dB')
    preProcess('-15dB')


