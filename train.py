import os, time
import pandas as pd
from utilies import data_load, training

def main(look_back_step, look_forward_step, train_data_path):
    start_time = time.time()
    mode = "s"
    item_dict = {'WATER_TEMPERATURE': 0, 'PH': 0, 'DISSOLVED_OXYGEN': 0, 'TURBIDITY': 0, 'CONDUCTIVITY': 0,
                 'PERMANGANATE_VALUE': 0, 'AMMONIA_NITROGEN': 0}
    for item, item_index in item_dict.items():
        print(item)
        main_path = look_back_step + "_" + look_forward_step + "/" + item+"/wm_tain"
        log_path = look_back_step + "_" + look_forward_step + "/" + item + "/wlog"
        save_name = "wm_repeat_train.csv"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        hidden_size_data = pd.read_csv(look_back_step + "_" + look_forward_step + "/" + item + "/w_tune/w_tune.csv", header=0,
                                       usecols=["output", "time_step", "hidden_size_1", "hidden_size_2"])

        hidden_size_list = hidden_size_data[hidden_size_data["output"] == item].loc[:, ["hidden_size_1", "hidden_size_2"]].values[0].tolist()
        train_data, valid_data, test_data = data_load(train_data_path,item,look_back_step, look_forward_step)
        time_step = look_back_step
        training(train_data, valid_data, test_data, item, item_index, time_step, hidden_size_list, main_path,
                         save_name , mode, look_forward_step)

    end_time = time.time()
    consuming_time = end_time - start_time
    print("*" * 50 + "Repeat training completed for all indicators!!\n Consuming" + str(consuming_time)+"*" * 50)

if __name__ == '__main__':
    look_back_step = 42
    look_forward_step = 18
    train_data_path = '/train_data.csv'
    main(look_back_step, look_forward_step, train_data_path)
