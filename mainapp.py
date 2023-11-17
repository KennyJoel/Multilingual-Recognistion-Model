import streamlit as st
import whisper
import tempfile
import os

def main():
    st.title("Multilingual speech recognition model by Kuragayala Kenny Joel")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3","wav","m4a"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        temp_file.close()

        st.audio(temp_file_path, format="audio/mp3", start_time=0)

        # Load Whisper model
        model = whisper.load_model("medium")

        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(temp_file_path)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # Print the recognized text
        st.subheader("Results:")
        st.write(f"Detected language: {detected_language}")
        st.write(f"Recognized text: {result.text}")

        # Delete the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
