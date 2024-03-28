import os
import time
import pandas as pd
from utilies import data_load, hyper_tuner

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    start_time = time.time()
    mode = "s"
    item_dict = {'WATER_TEMPERATURE': 0, 'PH': 0, 'DISSOLVED_OXYGEN': 0, 'TURBIDITY': 0, 'CONDUCTIVITY': 0,
                 'PERMANGANATE_VALUE': 0, 'AMMONIA_NITROGEN':0}
    data_path = "三皇庙.csv"

    try:
        for look_forward in [2, 6, 18, 42]:
            print(look_forward)
            for look_back in [6, 18, 42, 90, 180]:
                print(look_back)
                j = -1
                results = pd.DataFrame()
                main_path = str(look_back) + "_" + str(look_forward) + "/w_tune"
                log_path = str(look_back) + "_" + str(look_forward) + "/wlog"
                save_name = "w_tune.csv"
                if not os.path.exists(main_path):
                    os.makedirs(main_path)
                if not os.path.exists(main_path):
                    os.makedirs(log_path)

                for item in item_dict:
                    j = j+1
                    train_data, valid_data, test_data, data_mean, data_std = data_load(data_path, item, look_back, look_forward)
                    print(item)
                    time_step = look_back
                    model_information = hyper_tuner(log_path, time_step, train_data, valid_data, test_data, item,
                                                    mode, look_forward)

                    model_information = pd.DataFrame(model_information, index=[j])
                    results = pd.concat([model_information, results])

                    information_path = os.path.join(main_path, save_name)
                    if not os.path.exists(information_path):
                        results.to_csv(information_path, encoding='utf-8', index=False)
                    else:
                        existing_data = pd.read_csv(information_path)
                        combined_data = pd.concat([existing_data, results], ignore_index=True)
                        combined_data.to_csv(information_path, index=False)

    except Exception as e:
        print(e)

    end_time = time.time()
    consuming_time = end_time - start_time
    print("*" * 50 + "所有指标重复训练完成!!\n耗时" + str(consuming_time)+"*" * 50 )

if __name__ == '__main__':
    main()
