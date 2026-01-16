from abc import abstractmethod
from typing import Any

from constants.segmenter_configurations import SegmenterConfig


class ParameterOption:
    """Configuration class for different parameter types."""

    def __init__(self, name: str, param_type: str, default_value: Any, **kwargs):
        self.name = name
        self.param_type = param_type  # 'boolean', 'continuous', 'categorical'
        self.default_value = default_value
        self.current_value = default_value

        # Type-specific configurations
        if param_type == "continuous":
            self.min_value = kwargs.get("min_value", 0.0)
            self.max_value = kwargs.get("max_value", 1.0)
            self.step = kwargs.get("step", 0.01)
            self.decimals = kwargs.get("decimals", 2)
        elif param_type == "categorical":
            self.options = kwargs.get("options", [])

        # Display configuration
        self.display_name = kwargs.get("display_name", name.replace("_", " ").title())
        self.description = kwargs.get("description", "")


def get_parameters_class(config: SegmenterConfig) -> type:
    """
    Returns the Parameters class for the given SegmenterConfig.
    Args:
        config (SegmenterConfig): The segmenter configuration enum.
    Returns:
        type: The Parameters class.
    """
    if config == SegmenterConfig.SILENCE:
        return SilenceParameters
    elif config == SegmenterConfig.HRPS:
        return HRPSParameters
    elif config == SegmenterConfig.CHROMAGRAM:
        return ChromagramParameters
    elif config == SegmenterConfig.MFCC:
        return MFCCParameters
    elif config == SegmenterConfig.TEMPOGRAM:
        return TempogramParameters
    elif config == SegmenterConfig.NC_COMBINATION:
        return NCCombinationParameters
    else:
        raise ValueError(f"Unknown configuration: {config}")


def get_parameters_options(config: SegmenterConfig) -> list[ParameterOption]:
    """
    Returns the list of ParameterConfig for the given SegmenterConfig.
    Args:
        config (SegmenterConfig): The segmenter configuration enum.
    Returns:
        list[ParameterConfig]: The list of parameter configurations.
    """

    return get_parameters_class(config).get_parameters_options()


class Parameters:
    @classmethod
    def from_dict(cls, params_dict: dict) -> "Parameters":
        import inspect

        kwargs = {}

        # Get the __init__ method signature
        init_signature = inspect.signature(cls.__init__)
        init_params = list(init_signature.parameters.keys())[1:]  # Skip 'self'

        for key, value in params_dict.items():
            # Check if the parameter exists in the __init__ method
            if key not in init_params:
                raise ValueError(f"Unknown parameter: {key} for class {cls.__name__}")
            kwargs[key] = value

        return cls(**kwargs)

    @staticmethod
    @abstractmethod
    def get_parameters_options() -> list[ParameterOption]:
        pass


class UndefinedParameters(Parameters):
    pass


class SilenceParameters(Parameters):
    def __init__(
        self,
        silence_type: str,
        min_silence_duration_sec: float,
        frame_length: int,
        hop_length: int,
        threshold: float,
        min_distance_sec: float,
    ):
        self.silence_type = silence_type
        self.min_silence_duration_sec = min_silence_duration_sec
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold = threshold
        self.min_distance_sec = min_distance_sec

    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        return [
            ParameterOption(
                name="silence_type",
                param_type="categorical",
                default_value="amplitude",
                options=["amplitude", "spectral"],
                display_name="Silence Type",
                description="Energy computation method for silence detection.\n\nIntuition: 'amplitude' uses RMS energy in time domain, efficient for clear silences like pauses between movements. 'spectral' uses STFT-based energy, more sensitive to subtle frequency changes, can detect quiet sustained notes or orchestral texture changes.",
            ),
            ParameterOption(
                name="min_silence_duration_sec",
                param_type="continuous",
                default_value=2.0,
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                decimals=2,
                display_name="Min Silence Duration (sec)",
                description="Minimum duration for valid silence regions.\n\nIntuition: Higher values reduce false positives by filtering brief pauses (like breathing), but may miss shorter boundaries. Lower values are more sensitive but risk over-segmentation from small pauses.",
            ),
            ParameterOption(
                name="frame_length",
                param_type="continuous",
                default_value=30000,
                min_value=4410,
                max_value=88200,
                step=1,
                display_name="Frame Length",
                description="Analysis window size for feature extraction, controlling temporal vs frequency resolution trade-off.\n\nIntuition: Larger frames provide better frequency resolution and smoother curves, ideal for harmonic analysis but with reduced temporal precision. Smaller frames offer better temporal precision for rapid changes but result in noisier curves with reduced frequency resolution.",
            ),
            ParameterOption(
                name="hop_length",
                param_type="continuous",
                default_value=30000,
                min_value=2205,
                max_value=44100,
                step=1,
                display_name="Hop Length",
                description="Step size between successive analysis windows, determining temporal granularity. Its value should logically be ≤ frame_length even though the computation is possible.\n\nIntuition: Smaller values provide finer temporal resolution but increase computational cost. Larger values create smoother curves and reduce computational load, but may miss brief changes.",
            ),
            ParameterOption(
                name="threshold",
                param_type="continuous",
                default_value=0.85,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Threshold",
                description="Peak detection sensitivity for potential transitions.\n\nIntuition: Higher values (≥0.7) detect only prominent peaks with fewer false positives, but may miss subtle boundaries. Lower values (≤0.4) provide more sensitive detection with higher recall, but generate more false positives.",
            ),
            ParameterOption(
                name="min_distance_sec",
                param_type="continuous",
                default_value=28.0,
                min_value=1.0,
                max_value=60.0,
                step=0.1,
                decimals=2,
                display_name="Minimum Distance (sec)",
                description="Minimum temporal separation between detected transitions, prevents multiple detections for same boundary.\n\nIntuition: Higher values ensure detection of distinct structural changes but may miss close sections. Lower values allow detection of rapid changes but risk over-segmentation.",
            ),
        ]


class HRPSParameters(Parameters):
    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        L_h_frames: int,
        L_p_bins: int,
        beta: float,
        downsampling_factor: int,
        threshold: float,
        min_distance_sec: float,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.L_h_frames = L_h_frames
        self.L_p_bins = L_p_bins
        self.beta = beta
        self.downsampling_factor = downsampling_factor
        self.threshold = threshold
        self.min_distance_sec = min_distance_sec

    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        return [
            ParameterOption(
                name="frame_length",
                param_type="continuous",
                default_value=2048,
                min_value=1024,
                max_value=8192,
                step=1,
                display_name="Frame Length",
                description="Analysis window size for feature extraction, controlling temporal vs frequency resolution trade-off.\n\nIntuition: Larger frames provide better frequency resolution and smoother curves, ideal for harmonic analysis but with reduced temporal precision. Smaller frames offer better temporal precision for rapid changes but result in noisier curves with reduced frequency resolution.",
            ),
            ParameterOption(
                name="hop_length",
                param_type="continuous",
                default_value=1024,
                min_value=512,
                max_value=4096,
                step=1,
                display_name="Hop Length",
                description="Step size between successive analysis windows, determining temporal granularity. Its value should logically be ≤ frame_length even though the computation is possible.\n\nIntuition: Smaller values provide finer temporal resolution but increase computational cost. Larger values create smoother curves and reduce computational load, but may miss brief changes.",
            ),
            ParameterOption(
                name="L_h_frames",
                param_type="continuous",
                default_value=51,
                min_value=1,
                max_value=501,
                step=2,
                display_name="Smoothing Window Size",
                description="Harmonic component temporal smoothing window, controls coherence of harmonic elements (must be odd).\n\nIntuition: Larger values (≥50) provide better isolation of sustained elements (notes/chords), but may blur rapid harmonic changes. Smaller values (≤20) preserve temporal detail but may introduce noise affecting separation quality.",
            ),
            ParameterOption(
                name="L_p_bins",
                param_type="continuous",
                default_value=51,
                min_value=1,
                max_value=501,
                step=2,
                display_name="Pitch Smoothing Size",
                description="Percussive component frequency smoothing filter, determines percussive isolation across spectrum (must be odd).\n\nIntuition: Larger values (≥100) capture broadband events like applause spanning multiple frequencies, but may merge distinct sources. Smaller values (≤50) preserve frequency resolution for isolated elements but may fragment broadband events.",
            ),
            ParameterOption(
                name="beta",
                param_type="continuous",
                default_value=1.8,
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                decimals=1,
                display_name="Beta",
                description="Separation strictness factor controlling harmonic/percussive/residual classification.\n\nIntuition: Higher β values create more selective classification, placing more content in residual component. Values around 2.0 are optimal for applause detection (mixed characteristics). Values >2.5 become too aggressive.",
            ),
            ParameterOption(
                name="downsampling_factor",
                param_type="continuous",
                default_value=5,
                min_value=1,
                max_value=50,
                step=1,
                display_name="Downsampling Factor",
                description="Factor by which to downsample the signal before processing, trading temporal resolution for computational speed.\n\nIntuition: Higher factors significantly speed up computation as the factor increases, but may lose fine temporal details. Lower factors preserve resolution but increase processing time.",
            ),
            ParameterOption(
                name="threshold",
                param_type="continuous",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Threshold",
                description="Peak detection sensitivity for potential transitions.\n\nIntuition: Higher values (≥0.7) detect only prominent peaks with fewer false positives, but may miss subtle boundaries. Lower values (≤0.4) provide more sensitive detection with higher recall, but generate more false positives.",
            ),
            ParameterOption(
                name="min_distance_sec",
                param_type="continuous",
                default_value=15.0,
                min_value=1.0,
                max_value=30.0,
                step=0.1,
                decimals=2,
                display_name="Minimum Distance (sec)",
                description="Minimum temporal separation between detected transitions, prevents multiple detections for same boundary.\n\nIntuition: Higher values ensure detection of distinct structural changes but may miss close sections. Lower values allow detection of rapid changes but risk over-segmentation.",
            ),
        ]


class NCParameters(Parameters):
    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        normalization_mode: str,  # '1', '2', 'max' or 'z'
        smooting_filter_length: int,
        downsampling_factor: int,
        log_compression_factor: float,
        ssm_smoothing_filter_length: int,
        ssm_smoothing_filter_direction: str,  # 0: forward; 1: backward; 2: both directions
        ssm_threshold: float,
        ssm_binarize: bool,  # Add binarize parameter
        nc_kernel_size: int,
        nc_variance: float,
        nc_smoothing_sigma: float,
        threshold: float,
        min_distance_sec: float,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.normalization_mode = normalization_mode
        self.smooting_filter_length = smooting_filter_length
        self.downsampling_factor = downsampling_factor
        self.log_compression_factor = log_compression_factor
        self.ssm_smoothing_filter_length = ssm_smoothing_filter_length
        self.ssm_smoothing_filter_direction = ssm_smoothing_filter_direction
        self.ssm_threshold = ssm_threshold
        self.ssm_binarize = ssm_binarize
        self.ssm_threshold = ssm_threshold
        self.nc_kernel_size = nc_kernel_size
        self.nc_variance = nc_variance
        self.nc_smoothing_sigma = nc_smoothing_sigma
        self.threshold = threshold
        self.min_distance_sec = min_distance_sec

    @staticmethod
    def get_base_parameters_options() -> list[ParameterOption]:
        """Base parameter options that can be overridden by subclasses"""
        return [
            ParameterOption(
                name="frame_length",
                param_type="continuous",
                default_value=4410,
                min_value=256,
                max_value=44100,
                step=256,
                display_name="Frame Length",
                description="Analysis window size for feature extraction, controlling temporal vs frequency resolution trade-off.\n\nIntuition: Larger frames provide better frequency resolution and smoother curves, ideal for harmonic analysis but with reduced temporal precision. Smaller frames offer better temporal precision for rapid changes but result in noisier curves with reduced frequency resolution.",
            ),
            ParameterOption(
                name="hop_length",
                param_type="continuous",
                default_value=2205,
                min_value=128,
                max_value=22050,
                step=128,
                display_name="Hop Length",
                description="Step size between successive analysis windows, determining temporal granularity. Its value should logically be ≤ frame_length even though the computation is possible.\n\nIntuition: Smaller values provide finer temporal resolution but increase computational cost. Larger values create smoother curves and reduce computational load, but may miss brief changes.",
            ),
            ParameterOption(
                name="normalization_mode",
                param_type="categorical",
                default_value="2",
                options=["1", "2", "max", "z"],
                display_name="Normalization Mode",
                description="Feature normalization method affecting how features are scaled.\n\nIntuition: '1' (L1 norm) preserves proportions, '2' (L2 norm) creates unit vectors (best for chromagrams), 'max' scales by peak value preserving dynamics (good for MFCCs), 'z' applies zero mean and unit variance standardization.",
            ),
            ParameterOption(
                name="smooting_filter_length",
                param_type="continuous",
                default_value=11,
                min_value=1,
                max_value=51,
                step=2,
                display_name="Smoothing Filter Length",
                description="Temporal smoothing for noise reduction and pattern enhancement (must be odd).\n\nIntuition: Longer filters (≥21) create smoother trajectories and reduce sensitivity to rapid fluctuations, but may miss brief transitions. Shorter filters (≤11) preserve temporal detail but may introduce noise in SSM computation.",
            ),
            ParameterOption(
                name="downsampling_factor",
                param_type="continuous",
                default_value=20,
                min_value=1,
                max_value=50,
                step=1,
                display_name="Downsampling Factor",
                description="Complexity reduction while preserving structural information.\n\nIntuition: Higher factors (≥20) provide significant computation and memory reduction for long recordings, but may lose fine temporal details. Lower factors (≤10) preserve temporal resolution but increase computational cost, ideal for precise transitions.",
            ),
            ParameterOption(
                name="log_compression_factor",
                param_type="continuous",
                default_value=1.0,
                min_value=0.0,
                max_value=50.0,
                step=1,
                decimals=1,
                display_name="Log Compression Factor",
                description="Gamma value for dynamic range compression, enhances weak components.\n\nIntuition: Higher values (≥10) apply stronger compression and emphasize subtle variations indicating boundaries. Lower values (≤5) preserve original dynamics and maintain contrasts but may miss subtle transitions.",
            ),
            ParameterOption(
                name="ssm_smoothing_filter_length",
                param_type="continuous",
                default_value=1,
                min_value=1,
                max_value=51,
                step=2,
                display_name="SSM Smoothing Filter Length",
                description="Smoothing filter for SSM noise reduction and diagonal structure enhancement (must be odd).\n\nIntuition: Longer filters create clearer block structures for musical segments but may merge close boundaries. Shorter filters preserve fine details but may introduce novelty curve noise.",
            ),
            ParameterOption(
                name="ssm_smoothing_filter_direction",
                param_type="categorical",
                default_value="both",
                options=["forward", "backward", "both"],
                display_name="SSM Smoothing Filter Direction",
                description="Temporal smoothing direction for the self-similarity matrix.\n\nIntuition: 'both' applies bidirectional smoothing (most common) with balanced results and enhanced structural patterns. 'forward' or 'backward' apply unidirectional smoothing, which may create asymmetric effects for specific structures.",
            ),
            ParameterOption(
                name="ssm_threshold",
                param_type="continuous",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="SSM Threshold",
                description="Similarity threshold for structural contrast enhancement.\n\nIntuition: Higher thresholds (≥0.7) create sparser SSMs with only strong similarities, providing clearer boundaries but may miss subtle transitions. Lower thresholds (≤0.3) include more similarity information but may introduce novelty curve noise.",
            ),
            ParameterOption(
                name="ssm_binarize",
                param_type="boolean",
                default_value=False,
                display_name="Binarize SSM",
                description="Convert thresholded SSM to binary (0/1) vs continuous values, creating sharper structural boundaries.\n\nIntuition: Binarization benefits chromagrams by providing sharp structural delineation. Continuous values are better for MFCC/tempogram as they preserve gradient information for optimal detection.",
            ),
            ParameterOption(
                name="nc_kernel_size",
                param_type="continuous",
                default_value=10,
                min_value=1,
                max_value=150,
                step=1,
                display_name="Novelty Curve Kernel Size",
                description="Gaussian kernel size for novelty detection, controls temporal window for structural changes.\n\nIntuition: Larger kernels (≥50) detect broader changes like major movements or sections but may miss local transitions. Smaller kernels (≤20) are sensitive to rapid changes but may create noisy curves with false positives.",
            ),
            ParameterOption(
                name="nc_variance",
                param_type="continuous",
                default_value=5.0,
                min_value=0.1,
                max_value=3.0,
                step=0.1,
                decimals=1,
                display_name="Novelty Curve Variance",
                description="Gaussian kernel spread parameter, controls detection window characteristics.\n\nIntuition: Higher variance (≥10) creates broader detection windows that emphasize gradual changes but reduce sensitivity to abrupt transitions. Lower variance (≤5) creates sharper detection windows better for precise moments but more susceptible to noise.",
            ),
            ParameterOption(
                name="nc_smoothing_sigma",
                param_type="continuous",
                default_value=10.0,
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                decimals=1,
                display_name="Novelty Curve Smoothing Sigma",
                description="Final novelty curve Gaussian smoothing to reduce noise and enhance peaks.\n\nIntuition: Higher sigma (≥15) produces very smooth curves with clear broad peaks for major boundaries but may merge close transitions. Lower sigma (≤5) preserves fine temporal details and rapid transitions but may retain noise affecting peak detection.",
            ),
            ParameterOption(
                name="threshold",
                param_type="continuous",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Threshold",
                description="Peak detection sensitivity for potential transitions.\n\nIntuition: Higher values (≥0.7) detect only prominent peaks with fewer false positives, but may miss subtle boundaries. Lower values (≤0.4) provide more sensitive detection with higher recall, but generate more false positives.",
            ),
            ParameterOption(
                name="min_distance_sec",
                param_type="continuous",
                default_value=15.0,
                min_value=1.0,
                max_value=120.0,
                step=0.5,
                decimals=2,
                display_name="Minimum Distance (sec)",
                description="Minimum temporal separation between detected transitions, prevents multiple detections for same boundary.\n\nIntuition: Higher values ensure detection of distinct structural changes but may miss close sections. Lower values allow detection of rapid changes but risk over-segmentation.",
            ),
        ]

    @classmethod
    def get_parameters_options_with_overrides(
        cls, overrides: dict | None = None
    ) -> list[ParameterOption]:
        """Get parameters with specific overrides for subclasses"""
        base_params = cls.get_base_parameters_options()

        if not overrides:
            return base_params

        # Create a copy and apply overrides
        result_params = []
        for param in base_params:
            if param.name in overrides:
                # Create new parameter with overridden values
                override_values = overrides[param.name]
                new_param = ParameterOption(
                    name=param.name,
                    param_type=param.param_type,
                    default_value=override_values.get(
                        "default_value", param.default_value
                    ),
                    **{
                        k: v for k, v in override_values.items() if k != "default_value"
                    },
                )
                # Copy original attributes that weren't overridden
                for attr in [
                    "min_value",
                    "max_value",
                    "step",
                    "decimals",
                    "options",
                    "display_name",
                    "description",
                ]:
                    if hasattr(param, attr) and attr not in override_values:
                        setattr(new_param, attr, getattr(param, attr))
                result_params.append(new_param)
            else:
                result_params.append(param)

        return result_params

    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        return NCParameters.get_parameters_options_with_overrides()


class ChromagramParameters(NCParameters):
    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        overrides = {
            "ssm_threshold": {"default_value": 0.7107271414888593},
            "nc_kernel_size": {
                "default_value": 7,
            },
            "ssm_binarize": {"default_value": True},
            "nc_variance": {"default_value": 2.5},
            "nc_smoothing_sigma": {"default_value": 9.995628886023805},
            "threshold": {"default_value": 0.240434864692009},
        }
        return NCParameters.get_parameters_options_with_overrides(overrides)


class MFCCParameters(NCParameters):
    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        overrides = {
            "ssm_threshold": {"default_value": 0.2380926071382548},
            "nc_kernel_size": {
                "default_value": 12,
            },
            "ssm_binarize": {"default_value": False},
            "nc_variance": {"default_value": 2.5},
            "nc_smoothing_sigma": {"default_value": 9.82692563784925},
            "threshold": {"default_value": 0.2075037010195877},
        }
        return NCParameters.get_parameters_options_with_overrides(overrides)


class TempogramParameters(NCParameters):
    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        overrides = {
            "ssm_threshold": {"default_value": 0.2108978291433163},
            "nc_kernel_size": {
                "default_value": 17,
            },
            "ssm_binarize": {"default_value": False},
            "nc_variance": {"default_value": 2.5},
            "nc_smoothing_sigma": {"default_value": 9.979441218266915},
            "threshold": {"default_value": 0.42662884598649076},
        }
        return NCParameters.get_parameters_options_with_overrides(overrides)


class NCCombinationParameters(Parameters):
    def __init__(
        self,
        chromagram_weight: float,
        mfcc_weight: float,
        tempogram_weight: float,
        combination_method: str,  # 'mean', 'max'
        threshold: float,
        min_distance_sec: float,
    ):
        self.chromagram_weight = chromagram_weight
        self.mfcc_weight = mfcc_weight
        self.tempogram_weight = tempogram_weight
        self.combination_method = combination_method
        self.threshold = threshold
        self.min_distance_sec = min_distance_sec

    @staticmethod
    def get_parameters_options() -> list[ParameterOption]:
        return [
            ParameterOption(
                name="chromagram_weight",
                param_type="continuous",
                default_value=0.9355057066672625,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Chromagram Weight",
                description="Importance of harmonic transitions (key changes, chord progressions, tonal shifts) in the combined curve.\n\nIntuition: The bigger this parameter is, the more harmonic features will count in the output curve. Higher weights emphasize tonal changes like key modulations and chord progressions, which are reliable indicators of opera section boundaries.",
            ),
            ParameterOption(
                name="mfcc_weight",
                param_type="continuous",
                default_value=0.8974752371086757,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="MFCC Weight",
                description="Importance of timbral changes (instrumentation, vocal texture, recording characteristics) in the combined curve.\n\nIntuition: The bigger this parameter is, the more timbral features will count in the output curve. Higher weights emphasize changes in instrumentation, vocal texture, and recording characteristics that often mark opera section boundaries.",
            ),
            ParameterOption(
                name="tempogram_weight",
                param_type="continuous",
                default_value=0.3787974413685181,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Tempogram Weight",
                description="Importance of rhythmic shifts (tempo changes, meter variations, applause transitions) in the combined curve.\n\nIntuition: The bigger this parameter is, the more rhythmic features will count in the output curve. Generally receives lower weights as opera transitions often maintain rhythmic continuity despite harmonic and timbral changes.",
            ),
            ParameterOption(
                name="combination_method",
                param_type="categorical",
                default_value="mean",
                options=["mean", "max"],
                display_name="Combination Method",
                description="Mathematical fusion approach for combining multiple novelty curves.\n\nIntuition: 'mean' computes weighted average requiring consensus across features, providing balanced results with reduced false positives but may miss single-feature transitions. 'max' takes maximum value emphasis, more sensitive with higher recall for diverse transitions but potentially more false positives.",
            ),
            ParameterOption(
                name="threshold",
                param_type="continuous",
                default_value=0.1245455819972867,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                decimals=2,
                display_name="Threshold",
                description="Peak detection sensitivity for potential transitions.\n\nIntuition: Higher values (≥0.7) detect only prominent peaks with fewer false positives, but may miss subtle boundaries. Lower values (≤0.4) provide more sensitive detection with higher recall, but generate more false positives.",
            ),
            ParameterOption(
                name="min_distance_sec",
                param_type="continuous",
                default_value=15.0,
                min_value=1.0,
                max_value=30.0,
                step=0.1,
                decimals=2,
                display_name="Minimum Distance (sec)",
                description="Minimum distance between detected transitions in seconds",
            ),
        ]
