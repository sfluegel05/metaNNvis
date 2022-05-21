from importlib import reload

import translations
import frameworks
from translations.Translation import Translation
from frameworks.Framework import Framework

TRANSLATIONS = [translations.Torch2TfTranslation]
FRAMEWORKS = [frameworks.PyTorchFramework, frameworks.TensorFlow2Framework]

def translate(model, to_framework, *args, **kwargs):

    input_key = ''
    for fw in frameworks:
        if isinstance(fw, Framework):
            if fw.is_framework_model(model):
                input_key = fw.get_framework_key()

    if input_key == '':
        print('Could not detect the model framework')
        return False

    for trans in TRANSLATIONS:
        if isinstance(trans, Translation):
            if input_key == trans.get_input() and to_framework == trans.get_output():
                return trans.translate(model, *args, **kwargs)

    print(f'Could not find a translation from {input_key} to {to_framework}')
    return False


