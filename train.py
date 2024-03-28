import os
import time
import pandas as pd
from utilies import data_load, hyper_tuner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(look_back_step, look_forward_step, train_data_path):
    start_time = time.time()
    mode = "s"
    item_dict = {'WATER_TEMPERATURE': 0, 'PH': 0, 'DISSOLVED_OXYGEN': 0, 'TURBIDITY': 0, 'CONDUCTIVITY': 0,
                 'PERMANGANATE_VALUE': 0, 'AMMONIA_NITROGEN':0}
    try:
        for item, item_index in item_dict.items():
            main_path = look_back_step + "_" + look_forward_step + "/" + item + "/w_tune"
            log_path = look_back_step+ "_" + look_forward_step + "/" + item + "/wlog"
            save_name = "w_tune.csv"
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            train_data, valid_data, test_data = data_load(train_data_path, item, look_back_step, look_forward_step)
            time_step = look_back_step
            results = pd.DataFrame()
            model_information = hyper_tuner(log_path, time_step, train_data, valid_data, test_data, item,
                                            item_index,
                                            mode, look_forward_step)
            results = results.append(model_information, ignore_index=True)
            results = results[['output', "time_step", 'hidden_size_1', 'hidden_size_2']]
            information_path = os.path.join(main_path, save_name)
            if not os.path.exists(information_path):
                results.to_csv(information_path, encoding='utf-8', index=False)
            else:
                results.to_csv(information_path, mode='a', header=False,   index=False)
                
    except Exception as e:
        print(e)

    end_time = time.time()
    consuming_time = end_time - start_time
    print("*" * 50 + "Repeat training completed for all indicators!!\n Consuming" + str(consuming_time)+"*" * 50 )

if __name__ == '__main__':
    look_back_step = 42
    look_forward_step = 18
    train_data_path = '/train_data.csv'
    main(look_back_step, look_forward_step, train_data_path)
