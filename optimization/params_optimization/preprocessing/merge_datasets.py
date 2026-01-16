import os

import numpy as np

out_file = "optimization/params_optimization/data/"

# Giulio Cesare
audio_id = "bar103-t2-c2"

# Hercules
# audio_id = "bar2-t1-c1"

# Judas Maccabaeus
# audio_id = "bar20-t2-c2"

feature_name = "tempo"
downsample_factor = 20

id = f"{feature_name}_ds-{downsample_factor}"

out_file = os.path.join(out_file, audio_id, id)

dataset_out = os.path.join(out_file, "dataset.npz")
if os.path.exists(dataset_out):
    print("Dataset already exists at ", dataset_out)
    exit(0)

# Combine datasets
segments_paths = os.listdir(out_file)
segments_paths = [
    os.path.join(out_file, f) for f in segments_paths if f.endswith(".npz")
]


combined_ssm = []
combined_t_ann_list = []
combined_sr = []
combined_from_seconds = []
combined_to_seconds = []
audio_ids = []

for i, segment_path in enumerate(segments_paths):
    print("Merging segment ", i + 1, "/", len(segments_paths))
    data = np.load(segment_path)
    ssm = data["ssm"]
    sr = data["sr"]
    t_ann_list = data["t_ann_list"]

    combined_ssm.append(ssm)
    combined_sr.append(sr)
    combined_t_ann_list.append(t_ann_list)
    combined_from_seconds.append(data["from_second"])
    combined_to_seconds.append(data["to_second"])
    audio_ids.append(data["audio_id"])

# Save combined dataset
np.savez(
    dataset_out,
    ssms=np.array(combined_ssm, dtype=object),
    srs=np.array(combined_sr),
    t_ann_lists=np.array(combined_t_ann_list, dtype=object),
    from_seconds=np.array(combined_from_seconds),
    to_seconds=np.array(combined_to_seconds),
    audio_ids=np.array(audio_ids, dtype=str),
)
