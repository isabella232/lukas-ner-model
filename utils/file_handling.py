import jsonlines
import pandas as pd


def write_output_to_file(output, path):
    with jsonlines.open(path, mode="w") as writer:
        for obj in output:
            writer.write(obj)


def write_df_to_file(df, path):
    json_form = df.to_json(orient="records", lines=True, force_ascii=False)

    with open(path, "w") as f:
        f.write(json_form)


def read_df_from_file(path):
    with jsonlines.open(path) as reader:
        obj_list = [obj for obj in reader]

    return pd.DataFrame(obj_list)
