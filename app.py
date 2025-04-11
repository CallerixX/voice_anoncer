import streamlit as st
import torch
from TTS.api import TTS
import soundfile as sf
import io
import os
import tempfile
import base64
import requests
from pydub import AudioSegment
import shutil
import json

# Инициализация TTS
@st.cache_resource
def load_tts():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

def convert_audio_format(input_path, output_format):
    audio = AudioSegment.from_wav(input_path)
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
        output_path = temp_file.name
        audio.export(output_path, format=output_format)
        return output_path

def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">Скачать {file_label}</a>'

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

def add_background_sound(voice_path, background_path, output_path, background_volume=0.3):
    voice = AudioSegment.from_wav(voice_path)
    background = AudioSegment.from_wav(background_path)
    
    # Нормализуем длину фонового звука под длину голоса
    if len(background) < len(voice):
        background = background * (len(voice) // len(background) + 1)
    background = background[:len(voice)]
    
    # Уменьшаем громкость фонового звука
    background = background - (20 * (1 - background_volume))
    
    # Смешиваем звуки
    combined = voice.overlay(background)
    combined.export(output_path, format="wav")

def load_voices():
    # Базовый список голосов
    voices = {
        "Мужские": {
            "Александр": "Александр",
            "Захар": "Захар",
            "Итан": "Итан",
            "Кирилл": "Кирилл",
            "Томас": "Томас"
        },
        "Женские": {
            "Алена": "Алена",
            "Елена": "Елена",
            "Катрин": "Катрин",
            "Мария": "Мария",
            "Светлана": "Светлана",
            "Камила": "Камила"
        }
    }
    
    # Загружаем дополнительные голоса из файла, если он существует
    if os.path.exists("voices/voices.json"):
        try:
            with open("voices/voices.json", "r", encoding="utf-8") as f:
                additional_voices = json.load(f)
                for gender in additional_voices:
                    if gender not in voices:
                        voices[gender] = {}
                    voices[gender].update(additional_voices[gender])
        except Exception as e:
            st.error(f"Ошибка при загрузке дополнительных голосов: {str(e)}")
    
    return voices

def save_voices(voices):
    # Сохраняем только пользовательские голоса
    user_voices = {
        "Мужские": {},
        "Женские": {}
    }
    
    # Базовые голоса, которые не нужно сохранять
    base_voices = {
        "Мужские": ["Александр", "Захар", "Итан", "Кирилл", "Томас"],
        "Женские": ["Алена", "Елена", "Катрин", "Мария", "Светлана", "Камила"]
    }
    
    # Фильтруем пользовательские голоса
    for gender in voices:
        for name, path in voices[gender].items():
            if name not in base_voices[gender]:
                user_voices[gender][name] = path
    
    # Сохраняем в файл
    with open("voices/voices.json", "w", encoding="utf-8") as f:
        json.dump(user_voices, f, ensure_ascii=False, indent=4)

def main():
    st.title("Озвучка текста")
    
    # Загрузка модели
    tts = load_tts()
    
    # Загрузка списка голосов
    voices = load_voices()
    
    # Функционал добавления нового голоса
    st.subheader("Добавить новый голос")
    uploaded_file = st.file_uploader("Загрузите аудиофайл", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            voice_name = st.text_input("Введите имя для голоса:")
        with col2:
            voice_gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="voice_gender_upload")
        
        if voice_name and st.button("Добавить голос"):
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Конвертируем в WAV
                output_path = os.path.join("voices", f"{voice_name}.wav")
                convert_to_wav(temp_path, output_path)
                
                # Добавляем голос в список
                voices[voice_gender][voice_name] = voice_name
                
                # Сохраняем обновленный список голосов
                save_voices(voices)
                
                st.success(f"Голос {voice_name} успешно добавлен!")
            except Exception as e:
                st.error(f"Ошибка при добавлении голоса: {str(e)}")
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    st.divider()
    
    # Выбор пола голоса
    gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="voice_gender_select")
    
    # Выбор конкретного голоса
    voice_name = st.selectbox("Выберите голос:", list(voices[gender].keys()))
    
    # Предпрослушка голоса
    if st.button("Предпрослушка голоса"):
        preview_text = f"Привет! Меня зовут {voice_name}, я могу озвучить твой текст."
        with st.spinner("Генерируем предпрослушку..."):
            # Создаем временный файл для предпрослушки
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Генерируем предпрослушку
                wav = tts.tts_to_file(
                    text=preview_text,
                    speaker_wav=f"voices/{voices[gender][voice_name]}.wav",
                    language="ru",
                    file_path=temp_path
                )
                st.audio(temp_path)
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Ввод текста для озвучки
    text = st.text_area("Введите текст для озвучки:", height=200)
    
    # Настройки голоса
    st.subheader("Настройки голоса")
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Скорость речи:", 0.5, 2.0, 1.0, 0.1)
    with col2:
        add_background = st.checkbox("Добавить фоновый звук")
    
    if add_background:
        background_file = st.file_uploader("Загрузите фоновый звук", type=['wav', 'mp3', 'ogg', 'm4a'])
        if background_file is not None:
            # Предпрослушка фонового звука
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as bg_temp_file:
                bg_temp_path = bg_temp_file.name
                # Сохраняем загруженный файл во временный файл
                with open(bg_temp_path, 'wb') as f:
                    f.write(background_file.getvalue())
                # Конвертируем в WAV
                convert_to_wav(bg_temp_path, bg_temp_path)
                st.caption("Предпрослушка фонового звука:")
                st.audio(bg_temp_path)
                os.unlink(bg_temp_path)
        
        background_volume = st.slider("Громкость фонового звука:", 0.0, 1.0, 0.3, 0.1)
    
    # Кнопка для озвучки
    if st.button("Озвучить текст") and text:
        with st.spinner("Генерируем аудио..."):
            # Создаем временный файл для результата
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Инициализируем пути для разных форматов
            mp3_path = None
            ogg_path = None
            
            try:
                # Генерируем аудио
                wav = tts.tts_to_file(
                    text=text,
                    speaker_wav=f"voices/{voices[gender][voice_name]}.wav",
                    language="ru",
                    file_path=temp_path
                )
                
                # Добавляем фоновый звук, если нужно
                if add_background and background_file is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as bg_temp_file:
                        bg_temp_path = bg_temp_file.name
                        # Сохраняем загруженный файл во временный файл
                        with open(bg_temp_path, 'wb') as f:
                            f.write(background_file.getvalue())
                        # Конвертируем в WAV
                        convert_to_wav(bg_temp_path, bg_temp_path)
                        # Добавляем фоновый звук
                        add_background_sound(temp_path, bg_temp_path, temp_path, background_volume)
                        os.unlink(bg_temp_path)
                
                # Воспроизводим аудио
                st.audio(temp_path)
                
                # Предлагаем скачать в разных форматах
                st.subheader("Скачать аудио")
                col1, col2, col3 = st.columns(3)
                
                try:
                    # Создаем временные файлы для разных форматов
                    mp3_path = convert_audio_format(temp_path, "mp3")
                    with col1:
                        st.markdown(get_binary_file_downloader_html(mp3_path, "MP3"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ошибка при конвертации в MP3: {str(e)}")
                
                with col2:
                    st.markdown(get_binary_file_downloader_html(temp_path, "WAV"), unsafe_allow_html=True)
                
                try:
                    ogg_path = convert_audio_format(temp_path, "ogg")
                    with col3:
                        st.markdown(get_binary_file_downloader_html(ogg_path, "OGG"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ошибка при конвертации в OGG: {str(e)}")
                
            finally:
                # Удаляем временные файлы
                for path in [p for p in [temp_path, mp3_path, ogg_path] if p is not None]:
                    if os.path.exists(path):
                        os.unlink(path)

if __name__ == "__main__":
    main() 