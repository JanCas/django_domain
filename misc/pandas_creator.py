import string
from pandas import read_csv, DataFrame
from django_domain.settings import BASE_DIR
from os.path import join
from numpy import array, sum, zeros, pad, floor, ceil, concatenate, flipud, shape
from re import findall
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from misc.helper_functions import duplicate_labels


def get_train_test_val(material_prop: string, augmentation: bool) -> dict:
    """
    gets the train/test/val data and puts them into a dataframe
    adds 2 coloumns to the dataframe 'chemical_form'
                                     'num_elem' which contains the number of elements added up
    :param material_prop: name of the folder in 'material_properties
    :return: {'train': train_df, 'test': test_df, 'val': val_df}
    """

    main_dir = join(BASE_DIR, 'data/material_properties', material_prop)

    test_df = read_csv(join(main_dir, 'test.csv'))
    test_df['chemical_form'] = test_df.apply(lambda x: x[0].split('_')[0], axis=1)
    test_df['num_elem'] = test_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                        axis=1)

    train_df = read_csv(join(main_dir, 'train.csv'))
    train_df['chemical_form'] = train_df.apply(lambda x: x[0].split('_')[0], axis=1)
    train_df['num_elem'] = train_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                          axis=1)

    val_df = read_csv(join(main_dir, 'val.csv'))
    val_df['chemical_form'] = val_df.apply(lambda x: x[0].split('_')[0], axis=1)
    val_df['num_elem'] = val_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                      axis=1)

    if augmentation:
        train_df = duplicate_labels(train_df)
        test_df = duplicate_labels(test_df)
        val_df = duplicate_labels(val_df)
    print('Labels Created')
    return {'train': train_df, 'test': test_df, 'val': val_df}


def find_input_size(x: dict) -> int:
    """
    finds the dimensions of the biggest matrix in any of the datasets
    This makes sure the right size is used when the model is compiled
    :param x:
    :return:
    """

    y_init = 0
    for elem in x.values():
        y_init = elem['num_elem'].max() if elem['num_elem'].max() > y_init else y_init

    return y_init


def get_train_test_val_X_vector(cbfv: string, y: dict, augmentation: bool) -> array:
    """
    reads the desired feature vector and pads it to the needed size to make it a trainable rank 2 tensor
    :param cbfv:
    :return:
    """
    dir = join(BASE_DIR, 'data/element_properties', f'{cbfv}.csv')
    train, test, val = y.values()
    chemical_form_train_list = train['chemical_form']
    chemical_form_test_list = test['chemical_form']
    chemical_form_val_list = val['chemical_form']

    X_df = read_csv(dir).T
    X_df.columns = X_df.iloc[0]
    X_df = X_df[1:]

    final_size = int(find_input_size(y))

    index = 0
    train_x_vector = zeros((len(train), final_size, len(X_df)))
    upside = True
    for form_train in chemical_form_train_list:
        try:
            train_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                                chem_form=form_train, upside=upside)
            if augmentation:
                upside = not upside
            index += 1
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_train))
            train = train.drop(form_train)
    train_x_vector = train_x_vector[:len(train)]
    print('forming the train x vector done')

    index = 0
    test_x_vector = zeros((len(test), final_size, len(X_df)))
    upside = True
    for form_test in chemical_form_test_list:
        try:
            test_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                               chem_form=form_test, upside=upside)
            if augmentation:
                upside = not upside
            index += 1
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_test))
            test = test.drop(form_test)
    test_x_vector = test_x_vector[:len(test)]
    print('forming the test x vector done')

    index = 0
    val_x_vector = zeros((len(val), final_size, len(X_df)))
    upside = True
    for form_val in chemical_form_val_list:
        try:
            val_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                              chem_form=form_val, upside=upside)
            if augmentation:
                upside = not upside
            index += 1
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_val))
            val = val.drop(form_val)
    val_x_vector = val_x_vector[:len(val)]
    print('forming the val x vector done')

    # reshaping the vectors into rank 4 vectors (samples, height, width, channels)
    train_x_vector = train_x_vector.reshape(train_x_vector.shape[0], train_x_vector.shape[1],
                                            train_x_vector.shape[2], 1)
    test_x_vector = test_x_vector.reshape(test_x_vector.shape[0], test_x_vector.shape[1],
                                          test_x_vector.shape[2], 1)
    val_x_vector = val_x_vector.reshape(val_x_vector.shape[0], val_x_vector.shape[1],
                                        val_x_vector.shape[2], 1)

    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print(f'Train shape -> {shape(train_x_vector)} || labels -> {len(train)}')
    print(f'Test shape  -> {shape(test_x_vector)}  || labels -> {len(test)}')
    print(f'Val shape   -> {shape(val_x_vector)}   || labels -> {len(val)}')
    print('--------------------------------------------------')
    print('--------------------------------------------------')

    return {'train': (train, train_x_vector), 'test': (test, test_x_vector), 'val': (val, val_x_vector)}


def get_x_with_chemical_formula(x_df: DataFrame, final_size: tuple, chem_form: string, upside: bool) -> array:
    """
    constructing one x matrix based on a chemical formulas and padding it in to the desired size
    :param x_df:
    :param final_size:
    :param chem_form:
    :return:
    """
    # creatin dictionary for the chemical formula
    form = dict(findall('([A-Z][a-z]?)([0-9]*\.?[0-9]+)', chem_form))
    form.update((k, int(ceil(float(v)))) for k, v in form.items())

    chem_form_array = zeros((int(sum(list(form.values()))), final_size[1]))

    row_index = 0
    for elem, amount in form.items():
        elem_np = x_df[elem].to_numpy()
        for i in range(amount):
            # create the x_vector
            chem_form_array[row_index] = elem_np
            row_index += 1

    # calculate the padding dimensions
    top_pad = int(floor((final_size[0] - chem_form_array.shape[0]) / 2.0))
    bottom_pad = int(ceil((final_size[0] - chem_form_array.shape[0]) / 2.0))

    chem_form_array = pad(chem_form_array, ((top_pad, bottom_pad), (0, 0)), constant_values=(0, 0))

    return chem_form_array if upside else flipud(chem_form_array)


def generate_image_data_generators(material_prop: string, cbfv: string, batch_size: int, augmentation: bool) -> dict:
    """
    generate the keras image preprocessing which will be used in the fit function
    :return:
    """
    y = get_train_test_val(material_prop, augmentation=augmentation)
    x = get_train_test_val_X_vector(cbfv=cbfv, y=y, augmentation=augmentation)

    # creating y_labels
    train_label = x['train'][0]['target'].to_numpy()
    test_label = x['test'][0]['target'].to_numpy()
    val_label = x['val'][0]['target'].to_numpy()

    image_gen = ImageDataGenerator()

    train_gen = image_gen.flow(x=x['train'][1], y=train_label, batch_size=batch_size)
    test_gen = image_gen.flow(x=x['test'][1], y=test_label, batch_size=batch_size)
    val_gen = image_gen.flow(x=x['val'][1], y=val_label, batch_size=batch_size)

    keras_input = Input(shape=(x['train'][1].shape[1], x['train'][1].shape[2], x['train'][1].shape[3]))
    return {'train': train_gen, 'test': test_gen, 'val': val_gen, 'input': keras_input}
