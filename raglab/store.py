"""SQLite persistence. Embeddings are stored as blobs so HNSW can be rebuilt with new parameters
without paying for embedding again."""
from __future__ import annotations
import json, sqlite3, threading, time, uuid
import numpy as np
from .config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, title TEXT, text TEXT, metadata TEXT, created REAL);
CREATE TABLE IF NOT EXISTS indexes (id TEXT PRIMARY KEY, name TEXT, config TEXT, status TEXT, stats TEXT,
                                    graph TEXT, created REAL);
CREATE TABLE IF NOT EXISTS chunks (index_id TEXT, ord INTEGER, doc_id TEXT, kind TEXT, parent INTEGER,
                                   s0 INTEGER, s1 INTEGER, text TEXT, context TEXT, metadata TEXT,
                                   embedding BLOB, PRIMARY KEY (index_id, kind, ord));
CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, data BLOB);
CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, index_id TEXT, query TEXT, config TEXT, trace TEXT, created REAL);
CREATE TABLE IF NOT EXISTS llmcache (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS presets (name TEXT PRIMARY KEY, config TEXT, created REAL);
CREATE TABLE IF NOT EXISTS qcache (id INTEGER PRIMARY KEY AUTOINCREMENT, index_id TEXT, cfg_hash TEXT, query TEXT,
                                   embedding BLOB, answer TEXT, created REAL);
CREATE TABLE IF NOT EXISTS sweeps (id TEXT PRIMARY KEY, evalset_id TEXT, label TEXT, spec TEXT, points TEXT, created REAL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS evalsets (id TEXT PRIMARY KEY, name TEXT, items TEXT, created REAL);
CREATE TABLE IF NOT EXISTS evalruns (id TEXT PRIMARY KEY, evalset_id TEXT, index_id TEXT, label TEXT, config TEXT,
                                     metrics TEXT, rows TEXT, created REAL);
-- chunks is keyed (index_id, kind, ord); looking a document up across indexes needs its own index,
-- or "which indexes still hold this document" scans the whole table once per document on a page.
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks (doc_id);
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, target TEXT, label TEXT, status TEXT,
                                 done INTEGER, total INTEGER, error TEXT, spec TEXT, started REAL, updated REAL);
-- A dataset a user named. It exists in its own right so an empty one survives a restart, before
-- anything has been uploaded into it. Membership itself is not kept here: a document belongs to
-- the collection named by its `metadata.corpus`, which is the one authoritative answer for sample
-- corpora, user datasets and documents that predate both. This table registers the names.
CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL);
-- Uniqueness is enforced here rather than only in the request handler, so two requests racing for
-- the same name cannot both win.
CREATE UNIQUE INDEX IF NOT EXISTS datasets_by_name ON datasets (lower(name));
"""

#: Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an existing database on
#: the old shape, so each one is applied on open. Additive only: a column is added with a NULL
#: default and readers treat NULL as "this run predates the column".
MIGRATIONS = [
    ("evalruns", "manifest", "TEXT"),
    ("sweeps", "manifest", "TEXT"),
]


class Store:
    def __init__(self, path=None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path or DATA_DIR / "raglab.sqlite"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> list[str]:
        """Bring an existing database up to the current shape. Returns the columns it added."""
        added = []
        for table, column, decl in MIGRATIONS:
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if have and column not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                added.append(f"{table}.{column}")
        if added:
            self.db.commit()
        return added

    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def x(self, sql: str, args=()):
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    def xmany(self, sql: str, rows):
        with self.lock:
            self.db.executemany(sql, rows)
            self.db.commit()

    # datasets
    def datasets(self) -> list[dict]:
        """Every user-created dataset, oldest first."""
        return self.q("SELECT id, name, created FROM datasets ORDER BY created")

    def dataset(self, did: str) -> dict | None:
        rows = self.q("SELECT id, name, created FROM datasets WHERE id=?", (did,))
        return rows[0] if rows else None

    def add_dataset(self, name: str) -> str:
        """Register a dataset name.

        :param name: The name, already trimmed and checked by the caller.
        :returns: The new dataset id.
        :raises sqlite3.IntegrityError: If the name is taken, case-insensitively.
        """
        did = uuid.uuid4().hex[:10]
        self.x("INSERT INTO datasets VALUES (?,?,?)", (did, name, time.time()))
        return did

    # documents
    def add_document(self, title: str, text: str, metadata: dict) -> str:
        did = uuid.uuid4().hex[:10]
        self.x("INSERT INTO documents VALUES (?,?,?,?,?)", (did, title, text, json.dumps(metadata), time.time()))
        return did

    def documents(self, with_text=False) -> list[dict]:
        cols = "id,title,metadata,created,length(text) AS chars" + (",text" if with_text else "")
        rows = self.q(f"SELECT {cols} FROM documents ORDER BY created")
        for r in rows:
            r["metadata"] = json.loads(r["metadata"] or "{}")
        return rows

    def update_document(self, did: str, title: str, metadata: dict) -> None:
        """Replace a document's title and metadata. Its text and id are untouched.

        :param did: Document id.
        :param title: The new title.
        :param metadata: The complete new metadata mapping.
        """
        self.x("UPDATE documents SET title=?, metadata=? WHERE id=?", (title, json.dumps(metadata), did))

    def document(self, did: str) -> dict | None:
        rows = self.q("SELECT * FROM documents WHERE id=?", (did,))
        if rows:
            rows[0]["metadata"] = json.loads(rows[0]["metadata"] or "{}")
        return rows[0] if rows else None

    # indexes
    def create_index(self, name: str, config: dict) -> str:
        iid = uuid.uuid4().hex[:8]
        self.x("INSERT INTO indexes VALUES (?,?,?,?,?,?,?)",
               (iid, name, json.dumps(config), "building", "{}", None, time.time()))
        return iid

    def indexes(self) -> list[dict]:
        rows = self.q("SELECT id,name,config,status,stats,created FROM indexes ORDER BY created DESC")
        for r in rows:
            r["config"], r["stats"] = json.loads(r["config"]), json.loads(r["stats"] or "{}")
        return rows

    def index(self, iid: str) -> dict | None:
        rows = self.q("SELECT * FROM indexes WHERE id=?", (iid,))
        if not rows:
            return None
        r = rows[0]
        r["config"], r["stats"] = json.loads(r["config"]), json.loads(r["stats"] or "{}")
        r["graph"] = json.loads(r["graph"]) if r["graph"] else None
        return r

    def delete_index(self, iid: str):
        self.x("DELETE FROM chunks WHERE index_id=?", (iid,))
        self.x("DELETE FROM indexes WHERE id=?", (iid,))
        self.x("DELETE FROM blobs WHERE key LIKE ?", (f"{iid}:%",))
        for f in DATA_DIR.glob(f"{iid}.*"):
            f.unlink()

    def put_blob(self, key: str, arr: np.ndarray):
        self.x("INSERT OR REPLACE INTO blobs VALUES (?,?)", (key, arr.astype(np.float32).tobytes()))

    def get_blob(self, key: str, dim: int) -> np.ndarray | None:
        rows = self.q("SELECT data FROM blobs WHERE key=?", (key,))
        return np.frombuffer(rows[0]["data"], dtype=np.float32).reshape(-1, dim) if rows else None
