import os

import psycopg2
from sqlalchemy import create_engine

# Engine создаём один раз на процесс. У SQLAlchemy Engine уже есть пул соединений —
# если создавать новый Engine на каждый вызов, пул не переиспользуется и каждый
# запрос открывает новое TCP+auth соединение к удалённой БД (лишние сотни мс).
_engine = None


def postgres_connection():
    global _engine

    if _engine is None:
        db_url = os.environ.get("STARTML_DATABASE_URL")
        if not db_url:
            raise RuntimeError("Переменная окружения STARTML_DATABASE_URL не задана.")

        try:
            _engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
        except Exception as e:
            print("❌ Ошибка при подключении к базе данных.")
            raise e

    return _engine
