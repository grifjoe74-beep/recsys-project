import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from database import postgres_connection
from fastapi import FastAPI
from loguru import logger
from schema import PostGet


# === Вспомогательные функции ===
def load_sql(query: str, dtypes: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Выполняет SQL-запрос через соединение postgres_connection и возвращает DataFrame.

    Аргументы:
        query: SQL-запрос
        dtypes: словарь типов колонок для pd.read_sql (по умолчанию None)

    Возвращает:
        pd.DataFrame с результатом запроса

    Исключения:
        RuntimeError, если произошла ошибка при выполнении запроса
    """
    conn = postgres_connection()

    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {query}") from e
    finally:
        # postgres_connection() отдаёт SQLAlchemy Engine — у него нет .close() (только .dispose()),
        # безусловный conn.close() падает с AttributeError на первом же запросе
        if hasattr(conn, "close"):
            conn.close()

    return df


def load_model(model_path: str = "model.pkl"):
    """
    Загружает ML-модель из pickle-файла.

    Если код запускается в LMS-окружении (IS_LMS=1),
    путь берётся из переменной окружения MODEL_PATH.
    Иначе используется локальный путь, переданный пользователем.

    Исключения:
        FileNotFoundError — если файл модели не найден.
        RuntimeError — если произошла ошибка при загрузке модели.
    """
    if os.environ.get("IS_LMS", "0") == "1":
        model_path = os.environ["MODEL_PATH"]

    logger.info(f"Загрузка модели из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Файл модели не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}") from e

    logger.success("Модель успешно загружена")

    return model


def load_features():
    """Загружаем признаки пользователей и постов из БД"""
    logger.info("Загрузка признаков пользователей...")

    user_dtypes = {
        "user_id": "uint32",
        "gender": "uint8",
        "age": "uint8",
        "country": "category",
        "city": "category",
        "os": "category",
        "source": "category",
        "exp_group": "uint8",
        "age_group": "category",
        "country_city_count": "uint16",
    }
    df_user_features = load_sql("select * from rec_user_features", dtypes=user_dtypes)

    logger.info("Загрузка признаков постов...")

    post_dtypes = {
        "post_id": "uint16",
        "topic": "string",
        "text": "string",
        "text_cluster": "uint8",
        "text_len": "uint32",
        "word_count": "uint16",
        "avg_word_len": "float16",
    }

    n_clusters = 16
    post_dtypes.update({f"dist_cluster_{i}": "float16" for i in range(n_clusters)})

    df_post_features = load_sql("select * from rec_post_features", dtypes=post_dtypes)

    logger.success(f"Загружено: users={df_user_features.shape}, posts={df_post_features.shape}")

    logger.info("Загрузка лайков пользователей...")

    query = "select * from rec_user_likes order by user_id"

    df_user_likes = load_sql(query)

    return df_user_features, df_post_features, df_user_likes


def extract_dt_features(dt: datetime) -> pd.DataFrame:
    """Формируем временные признаки из datetime"""
    dt_series = pd.Series([dt])
    features = pd.DataFrame(index=dt_series.index)

    features["year"] = dt_series.dt.year.astype("uint16")
    features["month"] = dt_series.dt.month.astype("category")
    features["day"] = dt_series.dt.day.astype("category")
    features["dayofweek"] = (dt_series.dt.dayofweek + 1).astype("category")
    features["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype("category")
    features["hour"] = dt_series.dt.hour.astype("category")

    # Категория времени суток
    part_of_day = pd.Series(
        pd.Categorical(
            [None] * len(dt_series),
            categories=["morning", "afternoon", "evening", "night"],
        ),
        index=dt_series.index,
    )
    part_of_day.loc[(dt_series.dt.hour >= 6) & (dt_series.dt.hour < 12)] = "morning"
    part_of_day.loc[(dt_series.dt.hour >= 12) & (dt_series.dt.hour < 18)] = "afternoon"
    part_of_day.loc[(dt_series.dt.hour >= 18) & (dt_series.dt.hour < 24)] = "evening"
    part_of_day.loc[(dt_series.dt.hour < 6)] = "night"
    features["part_of_day"] = part_of_day.astype("category")

    return features


# === Загрузка основных ресурсов ===
logger.info("Инициализация сервиса...")

# Создаём объект FastAPI
app = FastAPI()

# Загружаем модель в память
model = load_model()

# Загружаем признаки пользователей и постов и лайки пользователей
df_user_features, df_post_features, df_user_likes = load_features()

# Фолбэк для пользователя, которого нет в df_user_features (например, совсем новый): типичные
# значения по всей таблице — без этого cross-join с пустым срезом роняет сервис на predict_proba.
_cold_user_values: Dict[str, Any] = {}
for _col in df_user_features.columns:
    if _col == "user_id":
        continue
    _series = df_user_features[_col]
    if pd.api.types.is_numeric_dtype(_series):
        _cold_user_values[_col] = _series.median()
    else:
        _mode = _series.mode()
        _cold_user_values[_col] = _mode.iloc[0] if not _mode.empty else ""

logger.success("Сервис успешно инициализирован")


# Эндпойнт для получения рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.

    Аргументы:
        user_id: идентификатор пользователя
        dt: дата и время запроса (для временных признаков)
        limit: максимальное количество постов в ответе

    Возвращает:
        Список объектов PostGet с топ-N постами
    """
    # множество ID уже пролайканных постов (у нового/неизвестного пользователя строки в таблице нет)
    liked_rows = df_user_likes.loc[df_user_likes["user_id"] == user_id, "liked_posts"]
    liked_posts = set(liked_rows.iloc[0]) if len(liked_rows) else set()

    # Фильтруем df_post_features
    mask_not_liked = ~df_post_features["post_id"].isin(liked_posts)
    df_post_candidates = df_post_features[mask_not_liked]

    # Если после фильтрации меньше limit, оставляем все посты
    if len(df_post_candidates) < limit:
        df_post_candidates = df_post_features

    # Формируем временные признаки
    dt_features = extract_dt_features(dt)

    # Признаки пользователя: если его нет в таблице (новый/неизвестный user_id) — берём фолбэк
    user_rows = df_user_features[df_user_features["user_id"] == user_id]
    if user_rows.empty:
        user_rows = pd.DataFrame([{**_cold_user_values, "user_id": user_id}])

    # Создаем итоговый датафрейм с признаками
    df_final = pd.merge(
        pd.merge(
            dt_features,
            user_rows,
            how="cross",
        ),
        df_post_candidates,
        how="cross",
    ).drop(columns=["user_id", "post_id", "text"])

    # Предсказание вероятности лайка
    like_probs = model.predict_proba(df_final)[:, 1]

    # Находим top-N
    top_idx = np.argpartition(-like_probs, limit - 1)[:limit]

    # Берём посты по индексу
    top_posts = df_post_candidates.iloc[top_idx]

    # Формируем список моделей PostGet
    recs = [
        PostGet(id=int(pid), text=str(text), topic=str(topic))
        for pid, text, topic in zip(top_posts["post_id"], top_posts["text"], top_posts["topic"])
    ]

    return recs
