from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401 — register models on Base before create_all

    Base.metadata.create_all(bind=engine)

    # create_all only creates missing TABLES, never alters an existing
    # one, so a database that predates a column keeps working until the
    # first query touches it and dies with "no such column".
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Add columns introduced after a database was first created.

    SQLAlchemy's create_all only creates missing TABLES; it will not alter an
    existing one. Without this, an installation that predates a new column
    keeps working until the first query touches it and then fails with
    "no such column", which is a confusing way to learn about a schema change.
    """
    from sqlalchemy import inspect, text

    wanted = {
        "extracted_fields": {
            # Sub-heading the field sits under, so the review screen can group
            # the way the document does.
            "section": "VARCHAR",
            # Gold-dataset section the field belongs to. `group` is reserved
            # in SQL, hence the prefix.
            "field_group": "VARCHAR",
        },
        "documents": {
            # The gold-shaped document and its results table.
            "structured_json": "JSON",
            "tests_json": "JSON",
        },
    }
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in existing_tables:
                continue
            have = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))
