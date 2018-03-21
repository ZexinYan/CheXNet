import pandas as pd

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def write(name, array, file):
    print(name, end=' ', file=file)
    for i in array[:-1]:
        print(i, end=' ', file=file)
    print(array[-1], file=file)


if __name__ == '__main__':
    entries = pd.read_csv('Data_Entry_2017.csv')


    def split_data(input_file, output_file):
        with open(input_file, 'r') as input:
            with open(output_file, 'w+') as output:
                for line in input:
                    line = line.split()[0]
                    label = []
                    for label_name in CLASS_NAMES:
                        if label_name in entries[entries['Image Index'] == line]['Finding Labels'].values[0]:
                            label.append(1)
                        else:
                            label.append(0)
                    write(line, label, output)
                    print(line)

    split_data('val.txt', 'val_data.txt')
    # split_data('train.txt', 'train_data.txt')
    # split_data('test.txt', 'test_data.txt')
