import json
from copy import deepcopy

def adjust_lexicon(file_name, new_file_name = '/media/jade/yi_Data/Data/20220110_attribute_lexicon.json'):
    with open (file_name, 'r') as f:
        lexicon = json.load(f)
    print(lexicon.keys())
    new_name = 'Permeability Impact_absorption Stability Durability Fasteners Midsole Heel Color Fit Shape Fixation'.split()
    new_lexicon = {}
    for name in new_name:
        if name not in new_lexicon:
            new_lexicon[name] = lexicon[name]

    for i, j in new_lexicon.items():
        if i == 'Impact_absorption':
            new_lexicon[i].extend(lexicon['Energy_Return'])
        elif i == 'Stability':
            new_lexicon[i].extend(lexicon['Flexibility'])
        elif i == 'Fasteners':
            new_lexicon[i].extend(lexicon['Traction'])
        elif i == 'Midsole':
            new_lexicon[i].extend(lexicon['Outsole'])
            new_lexicon[i].extend(lexicon['Insole'])
        elif i == 'Heel':
            new_lexicon[i].extend(lexicon['Cushion'])
        elif i == 'Shape':
            new_lexicon[i].extend(lexicon['Upper'])
        elif i == 'Fit':
            new_lexicon[i].extend(lexicon['Weight'])
        elif i == 'Fixation':
            new_lexicon[i].extend(lexicon['Toe_Box'])
            new_lexicon[i].extend(lexicon['Collar'])
            new_lexicon[i].extend(lexicon['Tongue'])

    with open (new_file_name, 'w') as f:
        json.dump(new_lexicon, f)

    return new_lexicon

def to_5 (file_name, new_file_name = '/media/jade/yi_Data/Data/20220111_attribute_lexicon.json'):
    with open (file_name, 'r') as f:
        lexicon = json.load(f)
    print(lexicon.keys())
    new_lexicon = deepcopy(lexicon)
    # for k in lexicon.keys():
    #     new_lexicon[k] = []

    for i, j in lexicon.items():
        if i == 'Stability':
            new_lexicon[i].extend(lexicon['Fasteners'])
            del new_lexicon['Fasteners']
        if i == 'Sole':
            new_lexicon[i].extend(lexicon['Heel'])
            new_lexicon[i].extend(lexicon['Fixation'])
            del new_lexicon['Heel']
            del new_lexicon['Fixation']
        if i == 'Color':
            new_lexicon[i].extend(lexicon['Shape'])
            del new_lexicon['Shape']

    with open (new_file_name, 'w') as f:
        json.dump(new_lexicon, f)

    print(new_lexicon.keys())










to_5('/media/jade/yi_Data/Data/20220110_attribute_lexicon.json')






















# adjust_lexicon('/media/jade/yi_Data/Data/20211117_attribute_lexicon.json')
