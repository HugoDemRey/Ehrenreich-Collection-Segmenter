import os
from typing import Type

import numpy as np

from app.src.audio.audio_file import AudioFile
from app.src.audio.signal import Signal
from app.src.audio_features.builders import (
    BuilderFromSignal,
    ChromagramBuilder,
    MFCCBuilder,
    SpectrogramBuilder,
    SSMBuilder,
    TempogramBuilder,
)
from app.src.audio_features.features import BaseFeature, SelfSimilarityMatrix
from app.src.io.ts_annotation import TSAnnotations

# Giulio Cesare
audio_id = "bar103-t2-c2"
audiofile = "data/CHC/Bar-103/Bar-103__Track2_Channel2.wav"

# Hercules
# audio_id = "bar2-t1-c1"
# audiofile = "data/HKB/Bar-2/BAR-2/Bar-2__Track1_Channel1.wav"

# Judas Maccabaeus
# audio_id = "bar20-t2-c2"
# audiofile = "data/HKB/Bar-20/BAR-20/Bar-20__Track2_Channel2.wav"


ground_truths = f"data/ground_truths/{audio_id}-timestamps.txt"
feature_name = "mfcc20"
downsample_factor = 20

builder_class: Type[BuilderFromSignal] = None
builder_kwargs = dict(hop_length=2205, frame_length=4410)


if feature_name == "spec":
    builder_class = SpectrogramBuilder
elif feature_name == "chroma":
    builder_class = ChromagramBuilder
elif feature_name.startswith("mfcc"):
    builder_class = MFCCBuilder
    builder_kwargs.update(dict(n_mfcc=int(feature_name[4:])))
elif feature_name == "tempo":
    builder_class = TempogramBuilder

builder: BuilderFromSignal = builder_class(**builder_kwargs)

out_file = "optimization/params_optimization/data/"
os.makedirs(out_file, exist_ok=True)
id = f"{feature_name}_ds-{downsample_factor}"
os.makedirs(os.path.join(out_file, audio_id, id), exist_ok=True)
out_file = os.path.join(out_file, audio_id, id)

signal: Signal = AudioFile(audiofile).load()
y_true = TSAnnotations.load_transitions_txt(ground_truths)
print(y_true)

window_size_seconds = 900  # 15 minutes
stride = 600  # 10 minutes

n_iter = int((signal.duration_seconds() - window_size_seconds) / stride) + 1


for i in range(n_iter):
    from_second = i * stride
    to_second = from_second + window_size_seconds

    print("Processing segment ", i + 1, "/", n_iter)
    print("From ", from_second, "s to ", to_second, "s")

    print(f"    * Extracting {feature_name} features")
    subsignal = signal.subsignal(from_second, to_second)
    feature: BaseFeature = builder.build(subsignal)

    print(f"    * Normalizing {feature_name} features")
    feature = feature.normalize(norm="2")
    print(f"    * Smoothing {feature_name} features")
    feature = feature.smooth(filter_length=11, window_type="boxcar")
    print(f"    * Downsampling {feature_name} features")
    feature = feature.downsample(factor=downsample_factor)
    # print(f"    * Log-compressing {feature_name} features")
    # feature = feature.ensure_positive()
    # feature = feature.log_compress(gamma=1)

    print(f"    * Computing SSM from {feature_name} features")
    ssm: SelfSimilarityMatrix = SSMBuilder(
        smoothing_filter_length=1,
        smoothing_filter_direction=2,
        shift_set=np.array([0]),
        tempo_relative_set=np.array([1]),
    ).build(feature)

    # Save plot of SSM for first segment only
    if i == 0:
        print("    * Saving SSM plot for first segment")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(ssm.data(), origin="lower", aspect="auto", cmap="gray_r")
        plt.title(f"SSM of {feature_name} features - Segment {i + 1}")
        plt.xlabel("Frames")
        plt.ylabel("Frames")
        plt.colorbar(label="Similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(out_file, f"segment_{i + 1}_{id}_ssm.png"))
        plt.close()

    print("    * Computing true boundaries for segment")
    y_true_seg = [t - from_second for t in y_true if t > from_second and t < to_second]

    # Save segment SSM and corresponding true boundaries

    print("    * Saving segment SSM and true boundaries")
    np.savez_compressed(
        os.path.join(out_file, f"segment_{i + 1}_{id}.npz"),
        ssm=ssm.data(),
        sr=ssm.sampling_rate(),
        t_ann_list=np.array(y_true_seg),
        from_second=from_second,
        to_second=to_second,
        audio_id=audio_id,
    )
