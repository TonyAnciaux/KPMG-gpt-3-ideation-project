import torch
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# TODO: this function needs a way to be prompted from a vocal keyword ("Hey KPMG"?)
# TODO: define dynamically the lengh of the recording (prompt-out keyword?)
def voice_record(duration, fs=16000):
    """
    Functions that triggers recording of sound input from user's microphone
    :param duration: the duration of the recording
    :param fs: samplerate of the audio, default value is 16000 as Facebook’s model only accepts this sampling rate
    :return: .wav file of the recording
    """
    fs = fs  # Sample rate
    seconds = duration  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('my-audio.wav', fs, myrecording)  # Save as WAV file


def speech_to_text(audio, fs=16000) -> str:
    """
    Function using state-of-the-art vocal recognition approach to convert audio recordings into written text
    :param audio: .wav file to be converted to text
    :param fs: samplerate of the audio, default value is 16000 as Facebook’s model only accepts this sampling rate
    :return: string variable
    """
    file_name = audio
    # If our sample rate is not 16000Hz, it'll convert it
    input_audio, _ = librosa.load(file_name, sr=fs)
    input_values = tokenizer(input_audio, return_tensors="pt", sampling_rate=fs).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print("This is the transcription:\n", transcription)


voice_record(5)
speech_to_text("my-audio.wav")
