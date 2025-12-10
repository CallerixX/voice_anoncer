import streamlit as st
import torch
import torchaudio

# Use the 'soundfile' backend for torchaudio to avoid optional torchcodec dependency
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    # ignore if backend can't be set; torchaudio will fall back to defaults
    pass

from TTS.api import TTS
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import os
import tempfile
import base64
import json
import time
import shutil

# --- КОНФИГУРАЦИЯ ---
# AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe" 

ST_PAGE_TITLE = "🎙️ AI Voice Studio Pro"
VOICES_DIR = "voices_pro"

# --- CSS И СТИЛЬ ---
def setup_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            font-weight: bold;
        }
        .stTextArea textarea {
            font-size: 18px;
            line-height: 1.5;
        }
        /* Подсветка для важных элементов */
        .highlight {
            padding: 10px;
            border-radius: 5px;
            background-color: #1e252b;
            border: 1px solid #333;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- БЭКЕНД: TTS ---
@st.cache_resource
def load_tts_model():
    """Загрузка модели XTTS v2. Кешируется для скорости."""
    original_load = torch.load
    # обход warning'а о weights_only в новых версиях torch
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Используем XTTS v2 - он лучший для RU в open-source на данный момент
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        return model
    except Exception as e:
        st.error(f"Критическая ошибка загрузки модели: {e}")
        return None
    finally:
        torch.load = original_load

# --- БЭКЕНД: УПРАВЛЕНИЕ ГОЛОСАМИ ---
class VoiceManager:
    def __init__(self, base_dir=VOICES_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_speakers(self):
        """Возвращает список доступных спикеров (папок)."""
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

    def get_styles(self, speaker_name):
        """Возвращает стили (файлы wav) для конкретного спикера."""
        speaker_path = os.path.join(self.base_dir, speaker_name)
        if not os.path.exists(speaker_path):
            return []
        return [f for f in os.listdir(speaker_path) if f.endswith(('.wav', '.mp3'))]

    def save_voice(self, speaker_name, style_name, audio_bytes, file_ext):
        """Сохраняет новый сэмпл голоса."""
        speaker_path = os.path.join(self.base_dir, speaker_name)
        os.makedirs(speaker_path, exist_ok=True)
        
        # Очищаем имя файла от мусора
        safe_style_name = "".join([c for c in style_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        filename = f"{safe_style_name}.wav" # Всегда сохраняем как wav для совместимости
        file_path = os.path.join(speaker_path, filename)

        # Конвертация любого входа в чистый WAV (mono, 22050Hz или 24000Hz оптимально для XTTS)
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            audio = AudioSegment.from_file(tmp_path)
            # Нормализация громкости референса
            audio = effects.normalize(audio)
            audio.export(file_path, format="wav")
            return True, "Голос успешно сохранен"
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def delete_style(self, speaker_name, style_filename):
        path = os.path.join(self.base_dir, speaker_name, style_filename)
        if os.path.exists(path):
            os.remove(path)
            # Если папка пуста, удаляем спикера
            if not os.listdir(os.path.join(self.base_dir, speaker_name)):
                os.rmdir(os.path.join(self.base_dir, speaker_name))

# --- БЭКЕНД: ОБРАБОТКА АУДИО ---
class AudioProcessor:
    @staticmethod
    def post_process_audio(input_path, output_path, remove_silence=True, normalize=True):
        """Улучшает синтезированное аудио."""
        audio = AudioSegment.from_wav(input_path)
        
        # 1. Удаление тишины в начале и конце
        if remove_silence:
            # Грубая обрезка тишины
            def match_target_amplitude(sound, target_dBFS):
                change_in_dBFS = target_dBFS - sound.dBFS
                return sound.apply_gain(change_in_dBFS)
            
            # Разбиваем по тишине и собираем обратно (более агрессивный метод)
            # Или простой strip_silence (менее рискованный)
            # Используем встроенный strip silence метод через pydub logic (manual implementation usually needed)
            # Для простоты: обрежем просто начало и конец если они тихие
            pass # Pydub не имеет простого .strip(), оставим как есть или добавим логику позже

        # 2. Нормализация
        if normalize:
            audio = effects.normalize(audio)

        audio.export(output_path, format="wav")

    @staticmethod
    def mix_background(voice_path, bg_path, output_path, bg_volume=0.2):
        """Накладывает музыку с приглушением."""
        voice = AudioSegment.from_wav(voice_path)
        bg = AudioSegment.from_file(bg_path)
        
        # Зацикливаем фон если он короче голоса
        if len(bg) < len(voice) + 1000: # +1 секунда хвоста
            loop_count = (len(voice) // len(bg)) + 2
            bg = bg * loop_count
            
        bg = bg[:len(voice) + 500] # Фон чуть длиннее голоса
        
        # Понижаем громкость фона
        bg = bg - (30 * (1 - bg_volume)) # Эвристическая формула громкости
        
        combined = voice.overlay(bg)
        combined.export(output_path, format="wav")

# --- UI КОМПОНЕНТЫ ---
def get_download_link(file_path, label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="text-decoration:none; background-color:#4CAF50; color:white; padding:8px 12px; border-radius:4px; font-weight:bold;">📥 Скачать {label}</a>'

# --- ГЛАВНАЯ ЛОГИКА ---
def main():
    st.set_page_config(page_title="AI Voice Studio", layout="wide", page_icon="🎙️")
    setup_style()
    
    st.title(ST_PAGE_TITLE)
    
    # Инициализация
    tts = load_tts_model()
    vm = VoiceManager()
    
    # Сайдбар с настройками
    with st.sidebar:
        st.header("Настройки генерации")
        
        speed = st.slider("Скорость речи", 0.5, 2.0, 1.1, 0.1, help="1.0 - норма. Для IVR лучше 1.1 - 1.2 (динамичнее).")
        temperature = st.slider("Эмоциональность (Temperature)", 0.01, 1.0, 0.75, 0.05, 
                                help="Низкая (0.1) - робот, стабильно. Высокая (0.8) - живо, но могут быть артефакты.")
        repetition_penalty = st.slider("Штраф за повторы", 1.0, 10.0, 2.0, 0.5, 
                                       help="Увеличьте, если голос начинает 'заедать' или повторять слоги.")
        
        st.divider()
        st.info("**Совет для IVR:** Для меню используйте скорость 1.1 и низкую вариативность (0.4). Для рекламы — скорость 1.0 и высокую вариативность (0.7+).")

    # Вкладки основного интерфейса
    tab_generate, tab_voices, tab_help = st.tabs(["Озвучка", "Лаборатория голосов", "Как пользоваться"])

    # --- Вкл 1: ОЗВУЧКА ---
    with tab_generate:
        col_settings, col_text = st.columns([1, 2])
        
        with col_settings:
            st.subheader("1. Выбор голоса")
            speakers = vm.get_speakers()
            
            if not speakers:
                st.warning("Нет сохраненных голосов! Перейдите в 'Лабораторию голосов' чтобы добавить.")
                selected_speaker = None
                selected_style = None
            else:
                selected_speaker = st.selectbox("Персонаж:", speakers)
                styles = vm.get_styles(selected_speaker)
                
                # Формируем список для отображения
                style_map = {f: f.replace('.wav', '') for f in styles}
                selected_style_file = st.selectbox(
                    "Эмоция / Стиль:", 
                    options=styles, 
                    format_func=lambda x: style_map[x]
                )
                
                if selected_style_file:
                    ref_path = os.path.join(VOICES_DIR, selected_speaker, selected_style_file)
                    st.audio(ref_path)
                    st.caption("Это референс, голос будет звучать похоже на него.")

            st.subheader("2. Фон (Опционально)")
            uploaded_bg = st.file_uploader("Музыка на фон", type=['mp3', 'wav'], key="bg_main")
            bg_vol = 0.2
            if uploaded_bg:
                bg_vol = st.slider("Громкость фона", 0.0, 1.0, 0.2)

        with col_text:
            st.subheader("3. Текст")
            text_input = st.text_area(
                "Введите текст для озвучки:", 
                height=300,
                placeholder="Здравствуйте! Вы позвонили в компанию Вектор. Нажмите один, чтобы связаться с оператором...",
                help="Используйте запятые для пауз. Цифры лучше писать словами для 100% точности ударений."
            )
            
            do_generate = st.button("СГЕНЕРИРОВАТЬ АУДИО", type="primary", disabled=(not text_input or not speakers))

        if do_generate:
            if not tts:
                st.error("Модель не загружена.")
            else:
                with st.status("Генерация аудио...", expanded=True) as status:
                    start_time = time.time()
                    
                    # Пути
                    ref_audio_path = os.path.join(VOICES_DIR, selected_speaker, selected_style_file)
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        output_path = temp_wav.name
                    
                    try:
                        # 1. Генерация
                        status.write("Синтез речи (нейросеть)...")
                        
                        # Предварительная обработка текста (простая)
                        # XTTS хорошо справляется с RU, но ударения можно форсировать символом '+' перед гласной в некоторых версиях, или используя '
                        
                        tts.tts_to_file(
                            text=text_input,
                            speaker_wav=ref_audio_path,
                            language="ru",
                            file_path=output_path,
                            speed=speed,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty
                        )
                        
                        # 2. Пост-обработка
                        status.write("Нормализация и обработка...")
                        AudioProcessor.post_process_audio(output_path, output_path)
                        
                        # 3. Наложение фона
                        final_path = output_path
                        if uploaded_bg:
                            status.write("Сведение с фоновой музыкой...")
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as bg_tmp:
                                bg_tmp.write(uploaded_bg.getvalue())
                                bg_tmp_path = bg_tmp.name
                            
                            mixed_path = output_path.replace(".wav", "_mixed.wav")
                            AudioProcessor.mix_background(output_path, bg_tmp_path, mixed_path, bg_volume=bg_vol)
                            final_path = mixed_path
                            os.unlink(bg_tmp_path)

                        status.update(label="Готово!", state="complete", expanded=False)
                        st.success(f"Сгенерировано за {time.time() - start_time:.2f} сек.")
                        
                        # Вывод результата
                        st.audio(final_path)
                        
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(get_download_link(final_path, "WAV (Лучшее качество)"), unsafe_allow_html=True)
                        
                        # Конвертация в MP3 для скачивания (легче вес)
                        mp3_path = final_path.replace(".wav", ".mp3")
                        AudioSegment.from_wav(final_path).export(mp3_path, format="mp3", bitrate="192k")
                        c2.markdown(get_download_link(mp3_path, "MP3 (Для веба)"), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    # --- Вкл 2: ЛАБОРАТОРИЯ ГОЛОСОВ ---
    with tab_voices:
        st.header("Управление банком голосов")
        st.markdown("""
        Здесь вы создаете **Клонов**. 
        Чтобы получить разные эмоции (Грусть, Радость, Строгость), загрузите соответствующие сэмплы для одного персонажа.
        """)
        
        col_new, col_list = st.columns([1, 2])
        
        with col_new:
            st.markdown("### Добавить новый сэмпл")
            new_speaker_name = st.text_input("Имя персонажа (например: Анна)", help="Группирует стили вместе")
            new_style_name = st.text_input("Название стиля (например: Приветливый)", help="Описание эмоции в сэмпле")
            uploaded_ref = st.file_uploader("Аудио-файл (WAV/MP3/OGG)", type=['wav', 'mp3', 'ogg', 'm4a'])
            
            if st.button("Сохранить голос"):
                if new_speaker_name and new_style_name and uploaded_ref:
                    file_ext = os.path.splitext(uploaded_ref.name)[1]
                    success, msg = vm.save_voice(new_speaker_name, new_style_name, uploaded_ref.read(), file_ext)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Заполните все поля!")

        with col_list:
            st.markdown("### Ваши голоса")
            current_speakers = vm.get_speakers()
            if not current_speakers:
                st.info("Пока нет добавленных голосов.")
            
            for spk in current_speakers:
                with st.expander(f"👤 {spk}", expanded=False):
                    styles = vm.get_styles(spk)
                    for stl in styles:
                        cols = st.columns([3, 1])
                        cols[0].write(f"🔹 {stl}")
                        if cols[1].button("🗑️", key=f"del_{spk}_{stl}"):
                            vm.delete_style(spk, stl)
                            st.rerun()
                    if not styles:
                        st.write("Нет стилей.")

    # --- Вкл 3: ПОМОЩЬ ---
    with tab_help:
        st.markdown("""
        ### Как добиться высокого качества для IVR?
        
        **1. Секрет эмоций (Cloning Strategy)**
        XTTS не понимает команду "скажи грустно". Он копирует интонацию из файла.
        * Хотите строгого оператора? -> Загрузите файл, где человек говорит строго. Назовите стиль "Строгий".
        * Хотите радостное приветствие? -> Загрузите файл с улыбкой в голосе. Назовите стиль "Радостный".
        
        **2. Рекомендации по тексту**
        * **Паузы:** Используйте длинное тире `—` или многоточие `...` для долгих пауз. Запятая `,` дает короткую паузу.
        * **Ударения:** Нейросеть обычно справляется, но если ошибается — попробуйте написать гласную большой буквой (пОезд) или использовать знак ' перед буквой.
        * **Числа:** Для IVR лучше писать "нажмите один", а не "нажмите 1". Это гарантирует правильное склонение.
        
        **3. Настройки**
        * **Скорость:** Для информационных сообщений ставьте 1.1. Для рекламы — 1.0.
        * **Temperature:** 0.6-0.7 — золотая середина. Меньше — голос станет монотонным (хорошо для диктовки номеров). Больше — эмоциональнее, но может "картавить".
        
        **4. Подготовка сэмпла**
        Загружайте чистый звук без шумов. Длительность от 6 до 10 секунд идеальна.
        """)

if __name__ == "__main__":
    main()