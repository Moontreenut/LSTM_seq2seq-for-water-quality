import os, time
import pandas as pd
from utilies import data_load, training


def main():
    start_time = time.time()
    mode = "s"
    item_dict = {'WATER_TEMPERATURE': 0, 'PH': 0, 'DISSOLVED_OXYGEN': 0, 'TURBIDITY': 0, 'CONDUCTIVITY': 0,
                 'PERMANGANATE_VALUE': 0, 'AMMONIA_NITROGEN':0}
    data_path = "三皇庙.csv"
    for look_forward in [2, 6, 18, 42]:
        for look_back in [6, 18, 42, 90, 180]:
            for item in item_dict:
                print(item)
                main_path = str(look_back) + "_" + str(look_forward) + "/" + item + "/wm_tain"
                save_name = "wm_repeat_train.csv"
                if not os.path.exists(main_path):
                    os.makedirs(main_path)

                hs_file_path = str(look_back) + "_" + str(look_forward) + "/w_tune/w_tune.csv"
                hidden_size_data = pd.read_csv(hs_file_path, header=0,
                                               usecols=["item", "time_step","look_forward", "hidden_size_1", "hidden_size_2"])

                hidden_size_list = hidden_size_data[hidden_size_data["item"] == item].loc[:, ["hidden_size_1", "hidden_size_2"]].values[0].tolist()

                train_data, valid_data, test_data, data_mean, data_std = data_load(data_path, item, look_back, look_forward)

                time_step = look_back
                training(train_data, valid_data, test_data, item, time_step, hidden_size_list, main_path,
                                 save_name, mode, look_forward, data_mean, data_std)

    end_time = time.time()
    consuming_time = end_time - start_time
    print("模型重复训练完成!!\n耗时" + str(consuming_time))

if __name__ == '__main__':
    main()