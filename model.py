#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################
import numpy as np
import torch
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt
from speechbrain.inference import SpeakerRecognition
import torch
from gtts import gTTS
import whisper


def anonymize(input_audio_path):
    """
    Anonymization pipeline combining signal processing and ML-based approaches with TTS.
    """
    # Étape 1 : Charger le fichier audio
    audio, sr = librosa.load(input_audio_path, sr=None)

    # Étape 2 : Prétraitement (normalisation, suppression de bruit, filtrage passe-bas)
    audio = normalize_amplitude(audio)
    audio = remove_noise(audio, sr)
    audio = apply_low_pass_filter(audio, sr)

    # Étape 3 : Extraire les embeddings x-vector
    xvector = extract_xvector(audio, sr)

    # Étape 4 : Anonymiser les embeddings
    anonymized_xvector = anonymize_xvector(xvector, noise_level=0.1)

    # Étape 5 : Transcrire le texte original avec Whisper
    try:
        original_text = transcribe_audio_whisper(input_audio_path)
    except Exception as e:
        print(f"Erreur lors de la transcription avec Whisper : {e}")
        original_text = "This is a sample text for evaluation."

    # Étape 6 : Générer une nouvelle voix anonymisée avec TTS
    temp_output_path = "temp_anonymized_audio.wav"
    
    # Initialiser le modèle TTS (exemple : Tacotron 2 avec vocoder Hifi-GAN)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Générer l'audio
    tts.tts_to_file(text=original_text, file_path=temp_output_path)

    # Étape 7 : Charger le nouvel audio anonymisé
    anonymized_audio, sr = librosa.load(temp_output_path, sr=None)

    # Étape 8 : Post-traitement (assurer la compatibilité avec soundfile.write)
    anonymized_audio = anonymized_audio.astype(np.float32)

    return anonymized_audio, sr


# Importation des fonctions déjà définies dans votre pipeline
def normalize_sampling_rate(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

def calculate_snr(audio):
    signal_power = np.sum(audio**2)
    noise_power = np.sum((audio - np.mean(audio))**2)
    return 10 * np.log10(signal_power / noise_power)

def normalize_amplitude(audio):
    return audio / np.max(np.abs(audio))

def remove_noise(audio, sr):
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def apply_low_pass_filter(audio, sr, cutoff_freq=3000):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def spectral_modification(audio, sr):
    fft_audio = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(fft_audio), 1 / sr)
    for i, freq in enumerate(frequencies):
        if 300 <= abs(freq) <= 2500:  # Bande critique pour les formants
            fft_audio[i] *= 0.7  # Réduire l'amplitude de ces fréquences
    modified_audio = np.real(np.fft.ifft(fft_audio))
    return modified_audio

def controlled_randomization(audio):
    noise = np.random.normal(0,0.0005, len(audio))
    randomized_audio = audio + noise
    return randomized_audio

def advanced_pseudonymization(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=28)
    modified_mfccs = mfccs + np.random.normal(0, 0.001, mfccs.shape)
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Initialiser le modèle x-vector
xvector_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def extract_xvector(audio, sr):
    # Extraire l'embedding x-vector
    embedding = xvector_model.encode_batch(torch.tensor(audio.copy()).unsqueeze(0)).squeeze().numpy()
    return embedding

def anonymize_xvector(embedding, noise_level=0.1):
    # Ajouter du bruit gaussien pour anonymiser l'embedding
    anonymized_embedding = embedding + np.random.normal(0, noise_level, embedding.shape)
    return anonymized_embedding

def transcribe_audio_whisper(audio_path):
    """
    Transcribes audio from the given file path using Whisper.
    """
    model = whisper.load_model("large")
    result = model.transcribe(audio_path)
    return result["text"]

def generate_anonymized_audio_with_gtts(text, output_path, lang='en'):
    """
    Generate anonymized audio using gTTS (Google Text-to-Speech).

    Parameters
    ----------
    text : str
        The text to be converted into speech.
    output_path : str
        Path where the anonymized audio will be saved.
    lang : str, optional
        Language of the text (default is 'en' for English).
    """
    # Initialiser gTTS avec le texte et la langue
    tts = gTTS(text=text, lang=lang)

    # Sauvegarder le fichier audio
    tts.save(output_path)