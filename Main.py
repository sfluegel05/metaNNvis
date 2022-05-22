from translations.Torch2TfTranslation import Torch2TfTranslation
from frameworks.PyTorchFramework import PyTorchFramework
from frameworks.TensorFlow2Framework import TensorFlow2Framework
from translations.Translation import Translation
from frameworks.Framework import Framework

TRANSLATIONS = [Torch2TfTranslation]
FRAMEWORKS = [PyTorchFramework, TensorFlow2Framework]


def translate(model, to_framework, *args, **kwargs):
    input_key = ''
    for fw in FRAMEWORKS:
        if isinstance(fw(), Framework):
            if fw.is_framework_model(model):
                input_key = fw.get_framework_key()

    if input_key == '':
        print('Could not detect the model framework')
        return False

    for trans in TRANSLATIONS:
        if isinstance(trans(), Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                return trans.translate(model, *args, **kwargs)

    print(f'Could not find a translation from {input_key} to {to_framework}')
    return False
