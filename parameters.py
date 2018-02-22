def config_select(config):
    if config[0] == 1 and config[1] == 0:
        model_name = 'single_lstm'
    elif config[0] == 1 and config[1] == 1:
        model_name = 'single_lstm_single_dense'
    elif config[0] == 2 and config[1] == 0:
        model_name = 'two_lstm'
    elif config[0] == 2 and config[1] == 1:
        model_name = 'two_lstm_single_dense'
    elif config[0] == 2 and config[1] == 2:
        model_name = 'two_lstm_two_dense'
    elif config[0] == 3 and config[1] == 0:
        model_name = 'three_lstm'
    elif config[0] == 3 and config[1] == 1:
        model_name = 'three_lstm_single_dense'
    elif config[0] == 3 and config[1] == 2:
        model_name = 'three_lstm_two_dense'
    elif config[0] == 3 and config[1] == 3:
        model_name = 'three_lstm_three_dense'
    else:
        raise ValueError

    return model_name