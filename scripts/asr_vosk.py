import sys
import sounddevice as sd
import queue
import vosk
import json
import curses
import textwrap
import shutil

print("Python executable:", sys.executable)

# Define the model path
model_path = "vosk-model-small-en-us-0.15"

# Initialize the Vosk model
model = vosk.Model(model_path)
q = queue.Queue()

def callback(indata, frames, time, status):
    """
    Callback function for the audio input stream.

    This function is called by the sounddevice library whenever new audio data is available. 
    It places the audio data into a queue for processing.

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
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def main(stdscr):
    """
    Main function to capture audio input, recognize speech using Vosk, and display the results in real-time.

    This function configures the audio input stream, initializes the Vosk speech recognizer, 
    and updates the terminal display with recognized text.

    Parameters
    ----------
    stdscr : _curses.window
        The curses window object representing the terminal screen.

    Returns
    -------
    None
    """
  
    # Clear screen
    stdscr.clear()
    
    # Configure the audio input
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                          channels=1, callback=callback):
        stdscr.addstr(0, 0, '#' * 80)
        stdscr.addstr(1, 0, 'Press Ctrl+C to stop the recording')
        stdscr.addstr(2, 0, '#' * 80)
        stdscr.refresh()

        rec = vosk.KaldiRecognizer(model, 16000)
        full_text = ""

        try:
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    final_text = json.loads(result)['text']
                    if final_text:
                        full_text += " " + final_text.strip()
                else:
                    # Optionally handle partial results here
                    partial_result = rec.PartialResult()
                    partial_text = json.loads(partial_result)['partial']
                    if partial_text:
                        display_text = full_text + " " + partial_text
                        # Get the width of the terminal
                        width = shutil.get_terminal_size().columns
                        # Wrap the text
                        wrapped_text = textwrap.wrap(display_text.strip(), width)
                        stdscr.clear()
                        stdscr.addstr(0, 0, '#' * 80)
                        stdscr.addstr(1, 0, 'Press Ctrl+C to stop the recording')
                        stdscr.addstr(2, 0, '#' * 80)
                        for i, line in enumerate(wrapped_text):
                            stdscr.addstr(3 + i, 0, line)
                        stdscr.refresh()
        except KeyboardInterrupt:
            stdscr.addstr(3 + len(wrapped_text), 0, "\nRecording stopped by user")
            stdscr.refresh()
            stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
