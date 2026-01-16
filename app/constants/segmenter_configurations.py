import enum

from src.audio.signal import Signal
from src.audio_features.builders import BuilderFromSignal


class SegmenterConfig(enum.Enum):
    UNDEFINED = -1
    SILENCE = 0
    HRPS = 1
    CHROMAGRAM = 2
    MFCC = 3
    TEMPOGRAM = 4
    NC_COMBINATION = 5


class SegmenterMVCBuilder:
    @staticmethod
    def get_model_controller(view, signal: Signal, config: SegmenterConfig, id: str):
        if (
            config == SegmenterConfig.CHROMAGRAM
            or config == SegmenterConfig.MFCC
            or config == SegmenterConfig.TEMPOGRAM
        ):
            from controllers.analysis_page.modules.segmenter_c import (
                SegmenterController,
            )
            from models.analysis_page.modules.segmenter_m import SegmenterModel

            model = SegmenterModel(signal, config, id)
            controller = SegmenterController(view, model)

        elif config == SegmenterConfig.SILENCE:
            from controllers.analysis_page.modules.segmenter_silence_c import (
                SegmenterSilenceController,
            )
            from models.analysis_page.modules.segmenter_silence_m import (
                SegmenterSilenceModel,
            )

            model = SegmenterSilenceModel(signal, id)
            controller = SegmenterSilenceController(view, model)

        elif config == SegmenterConfig.HRPS:
            from controllers.analysis_page.modules.segmenter_hrps_c import (
                SegmenterHrpsController,
            )
            from models.analysis_page.modules.segmenter_hrps_m import SegmenterHrpsModel

            model = SegmenterHrpsModel(signal, id)
            controller = SegmenterHrpsController(view, model)

        else:
            raise ValueError(f"Unknown controller configuration: {config}")

        return model, controller

    @staticmethod
    def get_builder(controller_config: SegmenterConfig) -> BuilderFromSignal:
        from src.audio_features.builders import (
            ChromagramBuilder,
            HRPSBuilder,
            MFCCBuilder,
            SilenceCurveBuilder,
            TempogramBuilder,
        )

        if controller_config == SegmenterConfig.CHROMAGRAM:
            return ChromagramBuilder()
        elif controller_config == SegmenterConfig.MFCC:
            return MFCCBuilder()
        elif controller_config == SegmenterConfig.TEMPOGRAM:
            return TempogramBuilder()
        elif controller_config == SegmenterConfig.SILENCE:
            return SilenceCurveBuilder()
        elif controller_config == SegmenterConfig.HRPS:
            return HRPSBuilder(51, 51, beta=2.0, frame_length=2048, hop_length=512)
        else:
            raise ValueError(f"Unknown controller configuration: {controller_config}")
