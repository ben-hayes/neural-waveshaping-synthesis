import time

import click
import gin
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from nts.models.timbre_transfer_newt import TimbreTransferNEWT
from nts.models.modules.shaping import FastNEWT

BUFFER_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

@click.command()
@click.option("--gin-file", prompt="Model config gin file")
@click.option("--output-file", prompt="output file")
@click.option("--num-iters", default=100)
@click.option("--batch-size", default=1)
@click.option("--device", default="cpu")
@click.option("--length-in-seconds", default=4)
@click.option("--use-fast-newt", is_flag=True)
@click.option("--model-name", default="ours")
def main(
    gin_file,
    output_file,
    num_iters,
    batch_size,
    device,
    length_in_seconds,
    use_fast_newt,
    model_name,
):
    gin.parse_config_file(gin_file)
    model = TimbreTransferNEWT()
    if use_fast_newt:
        model.newt = FastNEWT(model.newt)
    model.eval()
    model = model.to(device)

    # eliminate any lazy init costs
    with torch.no_grad():
        for i in range(10):
            model(
                torch.rand(4, 1, 250, device=device),
                torch.rand(4, 2, 250, device=device),
            )

    times = []
    with torch.no_grad():
        for bs in BUFFER_SIZES:
            dummy_control = torch.rand(
                batch_size,
                2,
                bs // 128,
                device=device,
                requires_grad=False,
            )
            dummy_f0 = torch.rand(
                batch_size,
                1,
                bs // 128,
                device=device,
                requires_grad=False,
            )
            for i in trange(num_iters):
                start_time = time.time()
                model(dummy_f0, dummy_control)
                time_elapsed = time.time() - start_time
                times.append(
                    [model_name, device if device == "cpu" else "gpu", bs, time_elapsed]
                )

    df = pd.DataFrame(times)
    df.to_csv(output_file)


if __name__ == "__main__":
    main()