import speech_recognition as sr
import moviepy.editor as mp

def transcribe_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Audio Transcription: ", text)
        return text
    except sr.UnknownValueError:
        print("Could not transcribe audio.")
    except sr.RequestError as e:
        print(f"Error with request: {e}")
