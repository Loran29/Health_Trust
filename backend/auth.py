from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, Field

DB_PATH = Path("backend/data/users.db")
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 7 * 24 * 3600  # 7 days

pwd_ctx = None  # unused, kept for reference
bearer = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())
router = APIRouter(prefix="/auth", tags=["auth"])


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                email          TEXT UNIQUE NOT NULL,
                password_hash  TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'user',
                created_at     INTEGER NOT NULL
            )
        """)


_init_db()


# ── Pydantic models ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=6)
    role: str = "user"


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict[str, Any]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_token(user_id: int, email: str, role: str) -> str:
    return jwt.encode(
        {
            "sub": str(user_id),
            "email": email,
            "role": role,
            "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict[str, Any]:
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "id": int(payload["sub"]),
            "email": payload["email"],
            "role": payload["role"],
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    role = req.role if req.role in ("user", "ngo") else "user"
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (req.email.lower(),)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (req.email.lower(), _hash_password(req.password), role, int(time.time())),
        )
        user_id = cur.lastrowid
    token = _make_token(user_id, req.email.lower(), role)
    return AuthResponse(
        token=token,
        user={"id": user_id, "email": req.email.lower(), "role": role},
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (req.email.lower(),)
        ).fetchone()
    if not row or not _verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _make_token(row["id"], row["email"], row["role"])
    return AuthResponse(
        token=token,
        user={"id": row["id"], "email": row["email"], "role": row["role"]},
    )


@router.get("/me")
def me(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return current_user
