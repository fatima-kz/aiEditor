import speech_recognition as sr
import moviepy.editor as mp

def transcribe_audio(video_path):
    try:
        video = mp.VideoFileClip(video_path)
    except FileNotFoundError:
        print(f"Video file not found: {video_path}")
        return None
    except Exception as e:
        print(f"Error loading video: {e}")
        return None

    # Check if the video has an audio track
    if not video.audio:
        print("The video does not contain an audio track.")
        return None

    audio_path = "temp_audio.wav"
    try:
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error writing audio to file: {e}")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        print("Audio Transcription: ", text)
        return text
    except sr.UnknownValueError:
        print("Could not transcribe audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with the speech recognition request: {e}")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
