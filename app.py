import streamlit as st
import torch
from TTS.api import TTS
from pydub import AudioSegment
import os
import tempfile
import base64
import json
import time

# Убедись, что путь к ffmpeg.exe указан верно
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe" # Или просто "ffmpeg", если он в PATH

@st.cache_resource
def load_tts():
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
    try:
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    finally:
        torch.load = original_load

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (находятся вне main)
def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href=\"data:application/octet-stream;base64,{b64}\" download=\"{os.path.basename(file_path)}\">Скачать {file_label}</a>'

def convert_audio_for_download(input_path, output_format):
    audio = AudioSegment.from_file(input_path)
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
        output_path = temp_file.name
        audio.export(output_path, format=output_format)
        return output_path

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
    
    os.makedirs("voices", exist_ok=True) 
    
    with open("voices/voices.json", "w", encoding="utf-8") as f:
        json.dump(user_voices, f, ensure_ascii=False, indent=4)

# ГЛАВНАЯ ФУНКЦИЯ ПРИЛОЖЕНИЯ
def main():
    st.title("Генератор голоса из текста")

    tts = load_tts()
    voices = load_voices()

    # --- Секция добавления нового голоса ---
    st.subheader("Добавить новый голос")
    uploaded_new_voice_file = st.file_uploader("Загрузите аудиофайл для нового голоса", type=['wav', 'mp3', 'ogg', 'm4a'], key="new_voice_upload")
    
    if uploaded_new_voice_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            voice_name = st.text_input("Введите имя для голоса:", key="new_voice_name")
        with col2:
            voice_gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="new_voice_gender_upload")
        
        if voice_name and st.button("Добавить голос"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_new_voice_file.name)[1]) as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_new_voice_file.getvalue())
            
            try:
                output_path_new_voice = os.path.join("voices", f"{voice_name}.wav")
                
                audio_for_new_voice = AudioSegment.from_file(temp_path, format=os.path.splitext(uploaded_new_voice_file.name)[1][1:])
                audio_for_new_voice.export(output_path_new_voice, format="wav")
                
                voices[voice_gender][voice_name] = voice_name
                save_voices(voices)
                st.success(f"Голос '{voice_name}' успешно добавлен!")
                st.rerun() 
            except Exception as e:
                st.error(f"Ошибка при добавлении голоса: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except PermissionError:
                        st.warning("Файл временно заблокирован. Попробуйте еще раз.")
    
    st.divider()

    # --- Секция выбора и предпрослушки голоса ---
    st.subheader("Выбор основного голоса")
    gender = st.radio("Выберите пол голоса:", ["Мужские", "Женские"], key="voice_gender_select")
    voice_name = st.selectbox("Выберите голос:", list(voices[gender].keys()))
    
    if st.button("Предпрослушка голоса"):
        preview_text = f"Привет! Меня зовут {voice_name}, я могу озвучить твой текст."
        with st.spinner("Генерируем предпрослушку..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as preview_file:
                preview_path = preview_file.name
                speaker_wav_file_for_preview = f"voices/{voices[gender][voice_name]}.wav" 
                
                tts.tts_to_file(
                    text=preview_text,
                    speaker_wav=speaker_wav_file_for_preview,
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
    
    st.divider()

    # --- Секция настроек голоса (скорость, вариативность, громкость, фоновый звук) ---
    st.subheader("Настройки голоса")
    col1, col2, col3 = st.columns(3)
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
            help="""Чем выше значение - тем разнообразнее интонации и акценты.
            Высокие значения (0.7-1.0) могут придать живости, но увеличивают риск заиканий и артефактов.
            Низкие значения (0.1-0.4) обеспечивают стабильность и повторяемость, но голос может звучать монотонно.
            Рекомендуем 0.5-0.7 для большинства задач."""
        )
    with col3:
        volume = st.slider(
            "Громкость голоса:",
            0.0, 2.0, 1.0, 0.1,
            help="Регулирует общую громкость синтезированной речи. 1.0 - стандартная."
        )
    
    add_background = st.checkbox("Добавить фоновый звук")
    background_data = None
    
    if add_background:
        background_file = st.file_uploader("Загрузите фоновый звук", type=['wav', 'mp3', 'ogg', 'm4a'], key="background_upload")
        if background_file is not None:
            uploaded_bg_extension = background_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_bg_extension}") as bg_temp_file:
                bg_temp_file.write(background_file.read())
                uploaded_bg_path = bg_temp_file.name
            
            try:
                converted_bg_path_for_processing = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                audio_bg = AudioSegment.from_file(uploaded_bg_path, format=uploaded_bg_extension)
                audio_bg.export(converted_bg_path_for_processing, format="wav")
                
                st.audio(converted_bg_path_for_processing, format="audio/wav")
                
                background_volume = st.slider(
                    "Громкость фонового звука:",
                    0.0, 1.0, 0.3, 0.1,
                    help="0 - фон отключен, 1 - максимальная громкость"
                )
                background_data = (converted_bg_path_for_processing, background_volume)
            except Exception as e:
                st.error(f"Ошибка обработки фонового звука: {str(e)}")
                background_data = None
            finally:
                if os.path.exists(uploaded_bg_path):
                    os.unlink(uploaded_bg_path)


    # --- Секция ввода текста и синтеза ---
    text = st.text_area(
        "Введите текст для озвучки:", 
        height=200,
        help="""Советы по форматированию для большей живости и стабильности:
1. Используйте '!' для повышения интонации и акцента.
2. Ставьте ',' для коротких пауз и естественного ритма.
3. Выделяйте ЗАГЛАВНЫМИ важные слова для усиления акцента.
4. Для ударения используйте символ ' перед ударной гласной: например, 'при́мер.
5. Для имитации эмоций и стиля, загружайте качественный образец голоса с нужной интонацией и эмоцией (длительность 3-6 секунд).
6. Если возникают заикания или пропуски слов, попробуйте изменить текст (перефразировать, упростить предложения) или уменьшить "Вариативность".
"""
    )
    
    final_speaker_wav_path = None

    st.subheader("Загрузите образец голоса для текущей озвучки (MP3, WAV, OGG, M4A)")
    st.info("""Это переопределит выбранный голос из списка для текущей озвучки.
            Для лучшего качества и имитации эмоций, используйте чистый, качественный аудиофайл (длительностью 3-6 секунд)
            с четкой, нейтральной или эмоционально окрашенной речью (в зависимости от желаемого эффекта).
            Избегайте фонового шума, эха и прерываний в образце.""")
    temp_speaker_audio_file = st.file_uploader("Загрузить аудиофайл для текущей озвучки", type=["mp3", "wav", "ogg", "m4a"], key="temp_speaker_upload")

    if temp_speaker_audio_file is not None:
        uploaded_file_extension = temp_speaker_audio_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file_extension}") as temp_audio_file_for_processing:
            temp_audio_file_for_processing.write(temp_speaker_audio_file.read())
            uploaded_audio_path_for_processing = temp_audio_file_for_processing.name

        st.audio(uploaded_audio_path_for_processing, format=f"audio/{uploaded_file_extension}")

        if uploaded_file_extension != "wav":
            st.info(f"Конвертируем {uploaded_file_extension.upper()} в WAV для обработки...")
            try:
                audio_to_convert = AudioSegment.from_file(uploaded_audio_path_for_processing, format=uploaded_file_extension)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file_for_processing:
                    final_speaker_wav_path = temp_wav_file_for_processing.name
                    audio_to_convert.export(final_speaker_wav_path, format="wav")
                st.success("Конвертация в WAV завершена.")
                os.unlink(uploaded_audio_path_for_processing)
            except Exception as e:
                st.error(f"Ошибка конвертации аудио: {e}")
                final_speaker_wav_path = None
        else:
            final_speaker_wav_path = uploaded_audio_path_for_processing
    else:
        final_speaker_wav_path = os.path.join("voices", f"{voices[gender][voice_name]}.wav")


    if st.button("Озвучить текст") and text:
        if final_speaker_wav_path and os.path.exists(final_speaker_wav_path):
            with st.spinner("Генерируем аудио..."):
                processed_text = (
                text
                .replace("'", "́")
                .replace("́ ", "́")
                )
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    output_synthesized_path = temp_path
                    final_output_path = temp_path
                    files_to_delete = [temp_path]

                    try:
                        tts.tts_to_file(
                            text=processed_text,
                            speaker_wav=final_speaker_wav_path,
                            language="ru",
                            file_path=output_synthesized_path,
                            speed=speed,
                            temperature=temperature
                        )
                        
                        synthesized_audio = AudioSegment.from_wav(output_synthesized_path)
                        delta_dB = 20 * (volume - 1.0)
                        synthesized_audio = synthesized_audio.apply_gain(delta_dB)
                        synthesized_audio.export(output_synthesized_path, format="wav")


                        if add_background and background_data is not None:
                            bg_path, bg_volume = background_data
                            final_output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                            add_background_sound(
                                output_synthesized_path, 
                                bg_path, 
                                final_output_path, 
                                background_volume=bg_volume
                            )
                            files_to_delete.append(final_output_path)
                            files_to_delete.append(bg_path)
                        
                        st.success("Синтез завершен!")
                        st.audio(final_output_path, format="audio/wav")
                        
                        st.subheader("Скачать в форматах:")
                        col1, col2, col3 = st.columns(3)
                        
                        try:
                            mp3_path = convert_audio_for_download(final_output_path, "mp3")
                            col1.markdown(get_binary_file_downloader_html(mp3_path, "MP3"), unsafe_allow_html=True)
                            files_to_delete.append(mp3_path)
                        except Exception as e:
                            st.error(f"Ошибка MP3: {str(e)}")
                        
                        col2.markdown(get_binary_file_downloader_html(final_output_path, "WAV"), unsafe_allow_html=True)
                        
                        try:
                            ogg_path = convert_audio_for_download(final_output_path, "ogg")
                            col3.markdown(get_binary_file_downloader_html(ogg_path, "OGG"), unsafe_allow_html=True)
                            files_to_delete.append(ogg_path)
                        except Exception as e:
                            st.error(f"Ошибка OGG: {str(e)}")
                        
                        time.sleep(2)

                    except Exception as e:
                        st.error(f"Произошла ошибка при синтезе: {e}")
                    finally:
                        for path in files_to_delete:
                            if path and os.path.exists(path):
                                try:
                                    os.unlink(path)
                                except Exception as e:
                                    st.error(f"Ошибка удаления файла {path}: {str(e)}")
                        
                        if temp_speaker_audio_file is not None and final_speaker_wav_path and os.path.exists(final_speaker_wav_path):
                            try:
                                os.unlink(final_speaker_wav_path)
                            except Exception as e:
                                st.error(f"Ошибка удаления временного файла образца голоса {final_speaker_wav_path}: {str(e)}")
        else:
            st.warning("Пожалуйста, загрузите образец голоса или выберите голос из списка.")

if __name__ == "__main__":
    os.makedirs("voices", exist_ok=True)
    main()
