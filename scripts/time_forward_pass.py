import time

import click
import gin
import numpy as np
import torch
from tqdm import trange

from nts.models.timbre_transfer_newt import TimbreTransferNEWT


@click.command()
@click.option("--gin-file", prompt="Model config gin file")
@click.option("--num-iters", default=100)
@click.option("--batch-size", default=1)
@click.option("--device", default="cpu")
@click.option("--length-in-seconds", default=4)
@click.option("--sample-rate", default=16000)
@click.option("--control-hop", default=128)
def main(
    gin_file, num_iters, batch_size, device, length_in_seconds, sample_rate, control_hop
):
    gin.parse_config_file(gin_file)
    dummy_control = torch.rand(
        batch_size,
        2,
        sample_rate * length_in_seconds // control_hop,
        device=device,
        requires_grad=False,
    )
    dummy_f0 = torch.rand(
        batch_size,
        1,
        sample_rate * length_in_seconds // control_hop,
        device=device,
        requires_grad=False,
    )
    model = TimbreTransferNEWT().to(device)
    model.eval()

    times = []
    with torch.no_grad():
        for i in trange(num_iters):
            start_time = time.time()
            model(dummy_f0, dummy_control)
            time_elapsed = time.time() - start_time
            times.append(time_elapsed)

    print(np.mean(times) / length_in_seconds)


if __name__ == "__main__":
    main()