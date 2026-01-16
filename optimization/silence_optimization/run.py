import hashlib
import json
import os

import numpy as np
import optuna
from tqdm import tqdm

from app.src.audio.audio_file import AudioFile
from app.src.io.ts_annotation import TSAnnotations
from app.src.metrics.metrics import TS_Evaluator
from optimization.silence_optimization.segmentation import (
    run_optimized_silence_segmentation,
)


def main():
    # Audio configurations - direct file paths
    audio_configs = [
        {
            "audio_id": "bar103-t2-c2",
            "audiofile": "data/CHC/Bar-103/Bar-103__Track2_Channel2.wav",
            "ground_truths": "data/ground_truths/bar103-t2-c2-timestamps.txt",
        },
        {
            "audio_id": "bar2-t1-c1",
            "audiofile": "data/HKB/Bar-2/BAR-2/Bar-2__Track1_Channel1.wav",
            "ground_truths": "data/ground_truths/bar2-t1-c1-timestamps.txt",
        },
        {
            "audio_id": "bar20-t2-c2",
            "audiofile": "data/HKB/Bar-20/BAR-20/Bar-20__Track2_Channel2.wav",
            "ground_truths": "data/ground_truths/bar20-t2-c2-timestamps.txt",
        },
    ]

    metric_name = "f1"

    # Initialize evaluator
    ts_evaluator: TS_Evaluator = TS_Evaluator(tolerance_seconds=15.0)
    if metric_name == "precision":
        metric_fn = ts_evaluator.precision
    elif metric_name == "recall":
        metric_fn = ts_evaluator.recall
    elif metric_name == "f1":
        metric_fn = ts_evaluator.f1_score
    elif metric_name == "f1_weighted":
        metric_fn = ts_evaluator.weighted_f1_score
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # Setup output directory
    id = f"silence_optimization_metric-{metric_name}"
    out_path = "optimization/silence_optimization/results"
    audio_ids = [config["audio_id"] for config in audio_configs]

    if len(audio_ids) == 1:
        out_path = os.path.join(out_path, audio_ids[0], id)
    else:
        hash_combined = "_".join(audio_ids)
        hash_combined = hashlib.md5(hash_combined.encode("utf-8")).hexdigest()[:8]
        out_path = os.path.join(out_path, f"combined-{hash_combined}", id)

    os.makedirs(out_path, exist_ok=True)

    # Find next available test folder number
    existing_tests = [
        d for d in os.listdir(out_path) if d.startswith("test") and d[4:].isdigit()
    ]
    if existing_tests:
        test_numbers = [int(d[4:]) for d in existing_tests]
        next_test_num = max(test_numbers) + 1
    else:
        next_test_num = 1

    # Create numbered test folder
    test_folder = f"test{next_test_num}"
    out_path = os.path.join(out_path, test_folder)
    os.makedirs(out_path, exist_ok=True)

    print(f"Results will be saved in: {out_path}")

    # Load audio files and ground truth annotations directly
    audio_data_list = []
    sampling_rates = []
    transitions_list = []
    audio_ids_list = []

    print("Loading audio files and annotations...")
    for config in audio_configs:
        print(f"Loading {config['audio_id']}...")

        # Load audio file
        audiofile_obj = AudioFile(config["audiofile"])
        signal = audiofile_obj.load()

        # Load ground truth annotations
        ts_annotations = TSAnnotations.load_transitions_txt(config["ground_truths"])

        # Store the entire audio file
        audio_data_list.append(signal.samples)
        sampling_rates.append(signal.sample_rate)
        transitions_list.append(np.array(ts_annotations))
        audio_ids_list.append(config["audio_id"])

        print(
            f"  - Duration: {signal.duration_seconds():.2f}s, Transitions: {len(ts_annotations)}"
        )

    print(f"Loaded {len(audio_configs)} audio files")
    print(f"Total transitions: {sum(len(t) for t in transitions_list)}")
    print("Starting Optuna study...")

    # Global best tracking
    best_metric = 0.0
    best_var = np.inf
    best_precision = 0.0
    best_recall = 0.0

    def objective(trial) -> float:
        nonlocal best_metric, best_var, best_precision, best_recall

        if best_metric == 1.0:
            return -best_metric  # Early stopping if perfect score is reached

        # Suggest parameters to optimize
        silence_type = trial.suggest_categorical(
            "silence_type", ["amplitude", "spectral"]
        )
        frame_length = trial.suggest_int("frame_length", 22050, 88200, step=4410)
        hop_length = trial.suggest_int("hop_length", 11025, 44100, step=2205)
        threshold = trial.suggest_float("threshold", 0.0, 1.0)
        min_silence_duration_sec = trial.suggest_float(
            "min_silence_duration_sec", 0.0, 5.0
        )
        min_distance_sec = trial.suggest_float("min_distance_sec", 15.0, 30.0)

        # Ensure hop_length <= frame_length
        if hop_length > frame_length:
            hop_length = frame_length

        # Compute segmentation on audio segments and evaluate metric
        metric_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []

        print(
            f"\n\n[Trial {trial.number}] Params: silence_type={silence_type}, frame_length={frame_length}, hop_length={hop_length}, threshold={threshold:.3f}, min_silence_duration={min_silence_duration_sec:.3f}, min_distance={min_distance_sec:.3f}"
        )

        # Store results for visualization
        silence_curves_sto = []
        peaks_secs_sto = []
        transitions_sto = []
        srs_sto = []

        # Progress bar for audio file processing
        audio_iter = zip(audio_data_list, sampling_rates, transitions_list)
        with tqdm(
            audio_iter,
            total=len(audio_data_list),
            desc=f"Trial {trial.number}",
            unit="audio",
            leave=False,
        ) as pbar:
            for i, (audio_data, sr, transitions) in enumerate(pbar):
                pbar.set_postfix(
                    {
                        "audio_id": audio_ids_list[i],
                        "duration": f"{len(audio_data) / sr:.0f}s",
                        "transitions": len(transitions),
                    }
                )

                try:
                    # Run silence segmentation on entire audio file
                    silence_curve, peaks_sec = run_optimized_silence_segmentation(
                        audio_data=audio_data,
                        sampling_rate=sr,
                        silence_type=silence_type,
                        frame_length=frame_length,
                        hop_length=hop_length,
                        threshold=threshold,
                        min_silence_duration_sec=min_silence_duration_sec,
                        min_distance_sec=min_distance_sec,
                    )

                    # Compute metric
                    metric_score: float = metric_fn(transitions, peaks_sec)
                    metric_scores.append(metric_score)

                    precisions.append(ts_evaluator.precision(transitions, peaks_sec))
                    recalls.append(ts_evaluator.recall(transitions, peaks_sec))

                    # Store for visualization (store all since we only have 3 audio files)
                    silence_curves_sto.append(silence_curve)
                    peaks_secs_sto.append(peaks_sec)
                    transitions_sto.append(transitions)
                    srs_sto.append(sr)

                except Exception as e:
                    pbar.set_postfix(
                        {"error": str(e)[:20] + "..." if len(str(e)) > 20 else str(e)}
                    )
                    # Use worst possible score for failed audio files
                    metric_scores.append(0.0)
                    precisions.append(0.0)
                    recalls.append(0.0)

        mean_metric = float(np.mean(metric_scores))
        var_metric = float(np.var(metric_scores))
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))

        print(
            f"\n\033[92m    * Mean {metric_name} score: {mean_metric:.4f}, Variance: {var_metric:.4f}\033[0m"
        )
        print(
            f"\033[94m    * Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}\033[0m"
        )

        # Save trial results
        trial_results = {
            "trial_number": trial.number,
            "datasets": audio_ids_list,
            "params": {
                "silence_type": silence_type,
                "frame_length": frame_length,
                "hop_length": hop_length,
                "threshold": threshold,
                "min_silence_duration_sec": min_silence_duration_sec,
                "min_distance_sec": min_distance_sec,
            },
            "metrics": {
                f"mean_{metric_name}": mean_metric,
                f"{metric_name}_variance": var_metric,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "individual_scores": metric_scores,
            },
        }

        # Save trial results to file
        trial_file = os.path.join(out_path, f"trial_{trial.number:03d}.json")
        with open(trial_file, "w") as f:
            json.dump(trial_results, f, indent=2)

        # Update best scores
        if mean_metric > best_metric or (
            mean_metric == best_metric and var_metric < best_var
        ):
            best_metric = mean_metric
            best_var = var_metric
            best_precision = mean_precision
            best_recall = mean_recall

            # Save best results
            best_results = {
                "best_trial": trial.number,
                "best_params": trial_results["params"],
                "best_metrics": trial_results["metrics"],
            }

            best_file = os.path.join(out_path, "best_results.json")
            with open(best_file, "w") as f:
                json.dump(best_results, f, indent=2)

            print(f"\033[93m    * NEW BEST! Trial {trial.number}\033[0m")

        # Return negative of mean metric for minimization
        return -mean_metric

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize")  # We return -metric, so minimize
    study.optimize(objective, n_trials=100, n_jobs=6)

    # Final results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {metric_name} score: {-study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save final study results
    final_results = {
        "study_summary": {
            "best_trial_number": study.best_trial.number,
            "best_value": -study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        },
        "all_trials": [],
    }

    for trial in study.trials:
        trial_info = {
            "number": trial.number,
            "value": -trial.value if trial.value is not None else None,
            "params": trial.params,
            "state": trial.state.name,
        }
        final_results["all_trials"].append(trial_info)

    final_file = os.path.join(out_path, "final_study_results.json")
    with open(final_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {out_path}")

    # Return optimization results
    return {
        "best_value": -study.best_value,
        "best_params": study.best_params,
        "precision": best_precision,
        "recall": best_recall,
        "variance": best_var,
        "n_trials": len(study.trials),
        "test_folder": test_folder,
        "out_path": out_path,
    }


if __name__ == "__main__":
    results = main()
    print(f"\n=== Final Results ===")
    print(f"Best {results.get('best_value', 'N/A'):.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"Results saved in: {results['out_path']}")
