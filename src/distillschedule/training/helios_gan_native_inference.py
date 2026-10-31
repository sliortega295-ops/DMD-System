"""W3 current Student output; shared video execution, independent GAN identity."""
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from . import helios_native_inference as common

SCHEMA = 'distillschedule.helios_gan_native_generator_inference.v1'
EXPORT_SCHEMA = 'distillschedule.helios_gan_native_output_request.v1'
FACTORY = 'distillschedule.training.helios_gan_native_inference:build_inference_model'


def inference_source():
    root = repository_root()
    return dict(shared=common.inference_source(), files={p: file_sha256(root / p) for p in (
        'src/distillschedule/training/helios_gan_native_inference.py',
        'src/distillschedule/training/helios_gan_training.py',
        'src/distillschedule/training/helios_gan_inputs.py')})


def _probe(parameters):
    return common._probe(parameters, schema=SCHEMA, variant='gan', source_factory=inference_source)


def fresh_noise_inputs(parameters):
    return common._fresh_noise_inputs(parameters, variant='gan')


def generate(state, feeds, options):
    return common.video._generate(state, feeds, options, prepare_noise=fresh_noise_inputs)


def refuse_training(*args, **kwargs):
    raise ValueError('portable W3 inference weights are not a training checkpoint')


def build_inference_model(parameters):
    return common._build_inference_model(parameters, probe=_probe, generate_callback=generate, refuse=refuse_training)


class NativeOutputRequest(common.NativeOutputRequest):
    export_schema = EXPORT_SCHEMA
    factory = FACTORY

    def project(self, view):
        return common._project(view, self.codec, self.value, schema=SCHEMA, variant='gan',
                               source_factory=inference_source, factory=FACTORY)


def export_native(config, checkpoint, output):
    from .config import TrainingConfig
    from .native import _admit, _helios_gan_stream
    from . import helios_gan_training as gan
    common.require(_helios_gan_stream(config, 'W3'), 'native GAN output requires its original full fresh-stream owner')
    raw = config.to_dict()
    request = raw['trainer'].pop('export', None)
    common.require(isinstance(request, dict), 'explicit W3 trainer.export codec/output policy required')
    raw['trainer'].update(checkpoint_every=0, resume_from=str(Path(checkpoint).expanduser().resolve()),
        output_dir=str(Path(output).expanduser().resolve().with_name(Path(output).name + '.native-owner')))
    clean = TrainingConfig.from_dict(raw, source=config.source)
    callback = NativeOutputRequest(request, output)
    def admit():
        result = _admit(clean)
        common.require(not result[1]['identity']['dirty'], 'clean committed source required')
        return result
    status = gan.run_training(clean, admit=admit, native_output=callback)
    common.require(status == 0 and callback.result is not None, 'original GAN owner did not publish requested output')
    return callback.result
