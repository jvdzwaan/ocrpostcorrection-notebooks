import json
from typing import Text

import pandas as pd
import typer
from typing_extensions import Annotated


def convert_to_icdar_output(
    predictions: Annotated[Text, typer.Option()],
    icdar_json: Annotated[Text, typer.Option()],
    out_file: Annotated[Text, typer.Option()],
) -> None:
    with open(icdar_json) as f:
        output = json.load(f)
    data = pd.read_csv(predictions, index_col=0)

    for _i, row in data.iterrows():
        output[row.text][row.token][row.pred] = 1.0

    with open(out_file, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    typer.run(convert_to_icdar_output)
