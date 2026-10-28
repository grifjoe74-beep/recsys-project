import os

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# один engine на процесс
_engine = None


def postgres_connection():
    global _engine

    if _engine is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise RuntimeError("Переменная окружения DATABASE_URL не задана.")

        try:
            _engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
        except Exception as e:
            print("❌ Ошибка при подключении к базе данных.")
            raise e

    return _engine
