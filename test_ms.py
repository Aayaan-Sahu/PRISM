from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import soundfile as sf
import urllib.request
import os

try:
    if not os.path.exists('mix.wav'):
        urllib.request.urlretrieve('https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/spex_plus_mix.wav', 'mix.wav')
    if not os.path.exists('ref.wav'):
        urllib.request.urlretrieve('https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/spex_plus_ref.wav', 'ref.wav')

    speaker_extraction = pipeline(Tasks.speaker_extraction, model='damo/speech_spex_plus_speaker-extraction_16k')
    res = speaker_extraction(( 'mix.wav', 'ref.wav' ))
    print(res.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
    print(e)
