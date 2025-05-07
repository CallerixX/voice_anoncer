import streamlit as st
import torch
from TTS.api import TTS
from pydub import AudioSegment
import os
import tempfile
import base64
import json
import time

AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

@st.cache_resource
def load_tts():
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
    try:
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    finally:
        torch.load = original_load

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
    
    if len(background) < len(voice):
        background = background * (len(voice) // len(background) + 1)
    background = background[:len(voice)]
    
    background = background - (20 * (1 - background_volume))
    combined = voice.overlay(background)
    combined.export(output_path, format="wav")

def load_voices():
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
    user_voices = {
        "Мужские": {},
        "Женские": {}
    }
    
    base_voices = {
        "Мужские": ["Александр", "Захар", "Итан", "Кирилл", "Томас"],
        "Женские": ["Алена", "Елена", "Катрин", "Мария", "Светлана", "Камила"]
    }
    
    for gender in voices:
        for name, path in voices[gender].items():
            if name not in base_voices[gender]:
                user_voices[gender][name] = path
    
    with open("voices/voices.json", "w", encoding="utf-8") as f:
        json.dump(user_voices, f, ensure_ascii=False, indent=4)

def main():
    st.title("Озвучка текста с настройками голоса")
    
    tts = load_tts()
    voices = load_voices()
    
    st.subheader("Добавить новый голос")
    uploaded_file = st.file_uploader("Загрузите аудиофайл", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            voice_name = st.text_input("Введите имя для голоса:")
        with col2:
            voice_gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="voice_gender_upload")
        
        if voice_name and st.button("Добавить голос"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())
            
            try:
                output_path = os.path.join("voices", f"{voice_name}.wav")
                convert_to_wav(temp_path, output_path)
                voices[voice_gender][voice_name] = voice_name
                save_voices(voices)
                st.success(f"Голос {voice_name} успешно добавлен!")
            except Exception as e:
                st.error(f"Ошибка при добавлении голоса: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except PermissionError:
                        st.warning("Файл временно заблокирован. Попробуйте еще раз.")
    
    st.divider()
    
    gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="voice_gender_select")
    voice_name = st.selectbox("Выберите голос:", list(voices[gender].keys()))
    
    if st.button("Предпрослушка голоса"):
        preview_text = f"Привет! Меня зовут {voice_name}, я могу озвучить твой текст."
        with st.spinner("Генерируем предпрослушку..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as preview_file:
                preview_path = preview_file.name
                tts.tts_to_file(
                    text=preview_text,
                    speaker_wav=f"voices/{voices[gender][voice_name]}.wav",
                    language="ru",
                    file_path=preview_path,
                    speed=1.0,
                    temperature=0.7
                )
                st.audio(preview_path)
                time.sleep(1)
                preview_file.close()
                try:
                    os.unlink(preview_path)
                except PermissionError:
                    pass
    
    st.subheader("Настройки голоса")
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider(
            "Скорость речи:", 
            0.5, 2.0, 1.0, 0.1,
            help="Чем выше значение - тем быстрее речь. Оптимальный диапазон: 0.8-1.2"
        )
    with col2:
        temperature = st.slider(
            "Вариативность:", 
            0.1, 1.0, 0.7, 0.1,
            help="Чем выше значение - тем разнообразнее интонации (но возможны артефакты). Рекомендуем 0.5-0.7"
        )
    
    add_background = st.checkbox("Добавить фоновый звук")
    background_data = None
    
    if add_background:
        background_file = st.file_uploader("Загрузите фоновый звук", type=['wav', 'mp3', 'ogg', 'm4a'])
        if background_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as bg_temp_file:
                bg_temp_path = bg_temp_file.name
                bg_temp_file.write(background_file.getvalue())
                
            try:
                converted_bg_path = "background.wav"
                convert_to_wav(bg_temp_path, converted_bg_path)
                st.audio(converted_bg_path)
                background_volume = st.slider(
                    "Громкость фонового звука:",
                    0.0, 1.0, 0.3, 0.1,
                    help="0 - фон отключен, 1 - максимальная громкость"
                )
                background_data = (converted_bg_path, background_volume)
            finally:
                try:
                    os.unlink(bg_temp_path)
                except Exception as e:
                    st.error(f"Ошибка удаления временного файла: {str(e)}")
    
    text = st.text_area(
        "Введите текст для озвучки:", 
        height=200,
        help="""Советы по форматированию:
1. Используйте ! для повышения интонации
2. Ставьте , для коротких пауз
3. Выделяйте ЗАГЛАВНЫМИ важные слова
4. Для ударения используйте символ ' перед буквой: напр. 'пример"""
    )
    
    if st.button("Озвучить текст") and text:
        with st.spinner("Генерируем аудио..."):
            processed_text = (
            text
            .replace("'", "́")  # Заменяем апостроф на акцент
            .replace("́ ", "́")  # Убираем пробелы после акцента
            )
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.close()
                
                mp3_path = None
                ogg_path = None
                final_path = temp_path
                files_to_delete = [temp_path]
                
                try:
                    tts.tts_to_file(
                        text=processed_text,
                        speaker_wav=f"voices/{voices[gender][voice_name]}.wav",
                        language="ru",
                        file_path=temp_path,
                        speed=speed,
                        temperature=temperature
                    )
                    
                    if add_background and background_data is not None:
                        bg_path, bg_volume = background_data
                        final_path = "final_output.wav"
                        add_background_sound(
                            temp_path, 
                            bg_path, 
                            final_path, 
                            background_volume=bg_volume
                        )
                        files_to_delete.append(final_path)
                    
                    st.audio(final_path)
                    
                    st.subheader("Скачать в форматах:")
                    col1, col2, col3 = st.columns(3)
                    
                    try:
                        mp3_path = convert_audio_format(final_path, "mp3")
                        col1.markdown(get_binary_file_downloader_html(mp3_path, "MP3"), unsafe_allow_html=True)
                        files_to_delete.append(mp3_path)
                    except Exception as e:
                        st.error(f"Ошибка MP3: {str(e)}")
                    
                    col2.markdown(get_binary_file_downloader_html(final_path, "WAV"), unsafe_allow_html=True)
                    
                    try:
                        ogg_path = convert_audio_format(final_path, "ogg")
                        col3.markdown(get_binary_file_downloader_html(ogg_path, "OGG"), unsafe_allow_html=True)
                        files_to_delete.append(ogg_path)
                    except Exception as e:
                        st.error(f"Ошибка OGG: {str(e)}")
                    
                    time.sleep(2)
                
                finally:
                    for path in files_to_delete:
                        if path and os.path.exists(path):
                            try:
                                os.unlink(path)
                            except Exception as e:
                                st.error(f"Ошибка удаления файла {path}: {str(e)}")
                    
                    if background_data is not None:
                        try:
                            os.unlink(background_data[0])
                        except Exception as e:
                            st.error(f"Ошибка удаления фонового файла: {str(e)}")

if __name__ == "__main__":
    os.makedirs("voices", exist_ok=True)
    main()
