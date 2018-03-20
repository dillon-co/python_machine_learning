import csv
import pdb
from dateutil.relativedelta import relativedelta
from datetime import datetime

def csv_as_lists_of_lists():
    lexicon = []
    with open('quakes.csv', ) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row_as_list = row[0].split(',')
            lexicon.append(row_as_list)
    return lexicon

def lexicon_sorted_by_years(lexicon):
    new_lexicon = []
    for row in lexicon:
        n_row_ind = ''
        if len(new_lexicon) > 1:
            for idx, n_row in enumerate(new_lexicon):
                if n_row[0][4] == row[4]:
                    n_row_ind = idx
        if n_row_ind == '':
            new_lexicon.append([row])
        else:
           new_lexicon[n_row_ind].append(row)
    new_lexicon.pop(0)
    return new_lexicon



def sorted_lexicon():
    lexicon = csv_as_lists_of_lists()
    year_lexicon = lexicon_sorted_by_years(lexicon)
    return year_lexicon

def sorted_lexicon_with_year_as_string():
    new_data_group = []
    y_l = sorted_lexicon()
    for year_group in y_l:
        new_year_group = []
        for quake in year_group:
            new_quake = []
            date_as_string = "-".join(quake[4:9])
            date_as_date = datetime.strptime(date_as_string, "%Y-%m-%d-%H-%M")
            new_quake.append(float(quake[0]))
            new_quake.append(date_as_date)
            [new_quake.append(q) for q in quake[11:14]]
            new_data_group.append(new_quake)
    return new_data_group

def reformatted_data():
    sorted_l = sorted_lexicon()
    new_l = sorted_lexicon_with_year_as_string()
    cronological_data = sorted(new_l, key=lambda quake: quake[1])
    return cronological_data

def inputs_with_outputs():
    ref_data = reformatted_data()
    for idx, quake in enumerate(ref_data):
        quake_warning = [0, 1]
        batch_size = idx+5
        for future_quake in ref_data[idx+1:batch_size]:
            if future_quake[0] > 4 and (future_quake[1] < (quake[1] + relativedelta(months=6))):
                quake_warning = [1, 0]
        quake.append(quake_warning)
    return ref_data

def create_featureset_and_labels():
    single_array_data = inputs_with_outputs()
    inputs = []
    outputs = []

    for indx, quake in enumerate(single_array_data):
        # print(indx, quake)
        if len(quake[4]) < 1 or quake[4][0] != '0':
            quake[4] = 0.482

        if quake[2][0] == '0':
            # print("meow")
            second_elem = float(quake[2])
            third_elem = float(quake[4])
        else:
            print("poop")
            second_elem = 0.01
            third_elem = float(quake[3])

        inputs.append([quake[0], second_elem, third_elem])
        outputs.append(quake[-1])

    print("Returning data")
    return inputs[0:2000], outputs[0:2000], inputs[2001:-1], outputs[2001:-1]

# create_featureset_and_labels()

# reformatted_data()
