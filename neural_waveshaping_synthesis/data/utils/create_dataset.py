import os
import shutil
from typing import Sequence

import gin
import numpy as np
from sklearn.model_selection import train_test_split

from .preprocess_audio import preprocess_audio
from ...utils import seed_all


def create_directory(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Failed to create directory %s" % path)
        else:
            print("Created directory %s..." % path)
    else:
        print("Directory %s already exists. Skipping..." % path)


def create_directories(target_root, names):
    create_directory(target_root)
    for name in names:
        create_directory(os.path.join(target_root, name))


def make_splits(
    audio_list: Sequence[str],
    control_list: Sequence[str],
    splits: Sequence[str],
    split_proportions: Sequence[float],
):
    assert len(splits) == len(
        split_proportions
    ), "Length of splits and split_proportions must be equal"

    train_size = split_proportions[0] / np.sum(split_proportions)
    audio_0, audio_1, control_0, control_1 = train_test_split(
        audio_list, control_list, train_size=train_size
    )
    if len(splits) == 2:
        return {
            splits[0]: {
                "audio": audio_0,
                "control": control_0,
            },
            splits[1]: {
                "audio": audio_1,
                "control": control_1,
            },
        }
    elif len(splits) > 2:
        return {
            splits[0]: {
                "audio": audio_0,
                "control": control_0,
            },
            **make_splits(audio_1, control_1, splits[1:], split_proportions[1:]),
        }
    elif len(splits) == 1:
        return {
            splits[0]: {
                "audio": audio_list,
                "control": control_list,
            }
        }


def lazy_create_dataset(
    files: Sequence[str],
    output_directory: str,
    splits: Sequence[str],
    split_proportions: Sequence[float],
):
    audio_files = []
    control_files = []
    audio_max = 1e-5
    means = []
    stds = []
    lengths = []
    control_mean = 0
    control_std = 1

    for i, (all_audio, all_f0, all_confidence, all_loudness, all_mfcc) in enumerate(
        preprocess_audio(files)
    ):
        file = os.path.split(files[i])[-1].replace(".wav", "")
        for j, (audio, f0, confidence, loudness, mfcc) in enumerate(
            zip(all_audio, all_f0, all_confidence, all_loudness, all_mfcc)
        ):
            audio_file_name = "audio_%s_%d.npy" % (file, j)
            control_file_name = "control_%s_%d.npy" % (file, j)

            max_sample = np.abs(audio).max()
            if max_sample > audio_max:
                audio_max = max_sample

            np.save(
                os.path.join(output_directory, "temp", "audio", audio_file_name),
                audio,
            )
            control = np.stack((f0, loudness, confidence), axis=0)
            control = np.concatenate((control, mfcc), axis=0)
            np.save(
                os.path.join(output_directory, "temp", "control", control_file_name),
                control,
            )

            audio_files.append(audio_file_name)
            control_files.append(control_file_name)

            means.append(control.mean(axis=-1))
            stds.append(control.std(axis=-1))
            lengths.append(control.shape[-1])

    if len(audio_files) == 0:
        print("No datapoints to split. Skipping...")
        return

    data_mean = np.mean(np.stack(means, axis=-1), axis=-1)[:, np.newaxis]
    lengths = np.stack(lengths)[np.newaxis, :]
    stds = np.stack(stds, axis=-1)
    data_std = np.sqrt(np.sum(lengths * stds ** 2, axis=-1) / np.sum(lengths))[
        :, np.newaxis
    ]

    print("Saving dataset stats...")
    np.save(os.path.join(output_directory, "data_mean.npy"), data_mean)
    np.save(os.path.join(output_directory, "data_std.npy"), data_std)

    splits = make_splits(audio_files, control_files, splits, split_proportions)
    for split in splits:
        for audio_file in splits[split]["audio"]:
            audio = np.load(os.path.join(output_directory, "temp", "audio", audio_file))
            audio = audio / audio_max
            np.save(os.path.join(output_directory, split, "audio", audio_file), audio)
        for control_file in splits[split]["control"]:
            control = np.load(
                os.path.join(output_directory, "temp", "control", control_file)
            )
            control = (control - data_mean) / data_std
            np.save(
                os.path.join(output_directory, split, "control", control_file), control
            )


@gin.configurable
def create_dataset(
    files: Sequence[str],
    output_directory: str,
    splits: Sequence[str] = ("train", "val", "test"),
    split_proportions: Sequence[float] = (0.8, 0.1, 0.1),
    lazy: bool = True,
):
    create_directories(output_directory, (*splits, "temp"))
    for split in (*splits, "temp"):
        create_directories(os.path.join(output_directory, split), ("audio", "control"))

    if lazy:
        lazy_create_dataset(files, output_directory, splits, split_proportions)

    shutil.rmtree(os.path.join(output_directory, "temp"))