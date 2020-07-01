import jsonlines
import pandas as pd


def write_df_to_file(df, path):
    json_form = df.to_json(orient="records", lines=True, force_ascii=False)
    f = open(path, "w")
    f.write(json_form)
    f.close()


def read_df_from_file(path):
    with jsonlines.open(path) as reader:
        obj_list = []
        for obj in reader:
            obj_list += [obj]
    return pd.DataFrame(obj_list)
