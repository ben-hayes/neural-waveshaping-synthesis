import os

import click
import gin
from scipy.io import wavfile
from tqdm import tqdm
import torch

from nts.data.urmp import URMPDataset
from nts.models.timbre_transfer_newt import TimbreTransferNEWT


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@click.command()
@click.option("--model-gin", prompt="Model .gin file")
@click.option("--model-checkpoint", prompt="Model checkpoint")
@click.option("--dataset-root", prompt="Dataset root directory")
@click.option("--dataset-split", default="test")
@click.option("--output-path", default="audio_output")
@click.option("--load-data-to-memory", default=False)
@click.option("--device", default="cuda:0")
@click.option("--batch-size", default=8)
@click.option("--num_workers", default=16)
def main(
    model_gin,
    model_checkpoint,
    dataset_root,
    dataset_split,
    output_path,
    load_data_to_memory,
    device,
    batch_size,
    num_workers,
):
    gin.parse_config_file(model_gin)
    make_dir_if_not_exists(output_path)

    data = URMPDataset(dataset_root, dataset_split, load_data_to_memory)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, num_workers=num_workers
    )

    device = torch.device(device)
    model = TimbreTransferNEWT.load_from_checkpoint(model_checkpoint).to(device)
    model.eval()

    for i, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            f0 = batch["f0"].float().to(device)
            control = batch["control"].float().to(device)
            output = model(f0, control)

        audio = output.cpu().numpy()
        for j in range(audio.shape[0]):
            file_name = "output_%d.wav" % (i * batch_size + j)
            wavfile.write(
                os.path.join(output_path, file_name), model.sample_rate, audio[j]
            )


if __name__ == "__main__":
    main()
