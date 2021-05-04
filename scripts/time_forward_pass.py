import time

import click
import gin
import numpy as np
from scipy.stats import describe
import torch
from tqdm import trange

from neural_waveshaping_synthesis.models.neural_waveshaping import NeuralWaveshaping
from neural_waveshaping_synthesis.models.modules.shaping import FastNEWT


@click.command()
@click.option("--gin-file", prompt="Model config gin file")
@click.option("--num-iters", default=100)
@click.option("--batch-size", default=1)
@click.option("--device", default="cpu")
@click.option("--length-in-seconds", default=4)
@click.option("--sample-rate", default=16000)
@click.option("--control-hop", default=128)
@click.option("--use-fast-newt", is_flag=True)
def main(
    gin_file, num_iters, batch_size, device, length_in_seconds, sample_rate, control_hop, use_fast_newt
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
    model = NeuralWaveshaping()
    if use_fast_newt:
        model.newt = FastNEWT(model.newt)
    model.eval()
    model = model.to(device)

    times = []
    with torch.no_grad():
        for i in trange(num_iters):
            start_time = time.time()
            model(dummy_f0, dummy_control)
            time_elapsed = time.time() - start_time
            times.append(time_elapsed)

    print(describe(times))
    rtfs = np.array(times) / length_in_seconds
    print("Mean RTF: %.4f" % np.mean(rtfs))
    print("90th percentile RTF: %.4f" % np.percentile(rtfs, 90))


if __name__ == "__main__":
    main()