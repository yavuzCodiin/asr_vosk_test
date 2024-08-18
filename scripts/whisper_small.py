import os
import sounddevice as sd
import queue
import threading
from scipy.io.wavfile import write
from transformers import pipeline

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize the transformers pipeline for speech-to-text
print("Initializing the transformer pipeline...")
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-small')
print("Transformer pipeline initialized.")

# Parameters
sample_rate = 16000
block_size = 6  # seconds
block_samples = int(sample_rate * block_size)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """
    Callback function to process audio data from the input stream.

    This function is called by the sounddevice library whenever a new block of audio data is available. 
    The audio data is placed into a queue for further processing.

    Parameters
    ----------
    indata : numpy.ndarray
        The recorded audio data.
    frames : int
        The number of frames of audio data.
    time : CData
        A structure containing timestamp information.
    status : sounddevice.CallbackFlags
        The status of the audio input stream.

    Returns
    -------
    None
    """
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

def transcribe_audio():
    """
    Function to transcribe audio data from the queue using the transformers pipeline.

    This function runs in a separate thread, continuously processing audio data from the queue. 
    Each audio block is saved to a temporary file, then transcribed using the speech-to-text pipeline.

    Returns
    -------
    None
    """
    print("Transcription thread started.")
    while True:
        audio_block = audio_queue.get()
        if audio_block is None:
            break
        # Save the audio block to a temporary file
        write('temp.wav', sample_rate, audio_block)
        # Transcribe the audio block
        print("Transcribing audio block...")
        text = whisper('temp.wav')
        recognized_text = text['text']
        print("Recognized Text:", recognized_text)

def start_recording():
    """
    Start recording and transcribing audio in real-time.

    This function starts the recording process, creating a new thread to handle audio transcription 
    while capturing audio data from the microphone.

    Returns
    -------
    None
    """
    print("Starting the recording...")
    threading.Thread(target=transcribe_audio, daemon=True).start()
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_samples):
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            audio_queue.put(None)
            print("Recording stopped by user")

if __name__ == "__main__":
    """
    Entry point of the script. Starts the audio recording and transcription process.
    """
    print("Running the script...")
    start_recording()
    print("Script finished.")
