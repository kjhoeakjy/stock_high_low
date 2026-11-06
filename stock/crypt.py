"""발행 데이터 암호화.

PAGE_PASSWORD 환경변수가 있으면 JSON·CSV를 `<이름>.enc`(AES-256-GCM, 키는 PBKDF2-SHA256 60만 회)로 저장하고
평문은 지운다. 없으면 평문 그대로. 화면(docs/index.html)은 같은 방식으로 브라우저에서 복호화한다.

  python -m stock.crypt docs/latest.json docs/history/*.json   # 기존 평문을 .enc로 변환(비밀번호 설정 직후 1회)
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITER = 1_000_000  # PBKDF2 반복(브라우저 1~3초). 무차별 대입 방어의 핵심은 비밀번호 길이(16자 이상 무작위 권장)
SALT = b"stock_high_low/v1"  # 고정 솔트: 브라우저가 키를 한 번만 유도하도록(비밀번호는 사용자 지정)
_KEYS: dict[str, bytes] = {}


def password() -> str | None:
    return os.environ.get("PAGE_PASSWORD") or None


def _key(pw: str) -> bytes:
    if pw not in _KEYS:
        _KEYS[pw] = PBKDF2HMAC(hashes.SHA256(), 32, SALT, ITER).derive(pw.encode())
    return _KEYS[pw]


def encrypt(data: bytes, pw: str) -> dict:
    iv = secrets.token_bytes(12)
    ct = AESGCM(_key(pw)).encrypt(iv, data, None)
    return {"v": 1, "kdf": "pbkdf2-sha256", "iter": ITER, "salt": SALT.decode(),
            "iv": base64.b64encode(iv).decode(), "ct": base64.b64encode(ct).decode()}


def decrypt(doc: dict, pw: str) -> bytes:
    return AESGCM(_key(pw)).decrypt(base64.b64decode(doc["iv"]), base64.b64decode(doc["ct"]), None)


def _enc_path(path: Path) -> Path:
    return path.with_name(path.name + ".enc")


def write_text(path: Path, text: str) -> Path:
    """비밀번호가 있으면 .enc로, 없으면 평문으로 저장. 반대쪽 파일은 지운다."""
    path = Path(path)
    pw = password()
    if pw:
        _enc_path(path).write_text(json.dumps(encrypt(text.encode(), pw)))
        if path.exists():
            path.unlink()
        return _enc_path(path)
    path.write_text(text)
    if _enc_path(path).exists():
        _enc_path(path).unlink()
    return path


def write_json(path: Path, obj) -> Path:
    return write_text(path, json.dumps(obj, ensure_ascii=False))


def read_text(path: Path) -> str | None:
    path = Path(path)
    if path.exists():
        return path.read_text()
    enc = _enc_path(path)
    pw = password()
    if enc.exists() and pw:
        return decrypt(json.loads(enc.read_text()), pw).decode()
    return None


def read_json(path: Path, default=None):
    t = read_text(path)
    return json.loads(t) if t is not None else default


def list_json(directory: Path) -> list[Path]:
    """평문·암호문을 합쳐 기준 경로(.enc 제외)를 정렬해 돌려준다."""
    directory = Path(directory)
    names = {p.name for p in directory.glob("*.json")} | {p.name[:-4] for p in directory.glob("*.json.enc")}
    return [directory / n for n in sorted(names)]


def migrate(paths: list[str]) -> None:
    pw = password()
    if not pw:
        print("[crypt] PAGE_PASSWORD 없음 → 평문 유지")
        return
    n = 0
    for s in paths:
        p = Path(s)
        if p.exists() and not p.name.endswith(".enc"):
            write_text(p, p.read_text())
            n += 1
    print(f"[crypt] {n}개 파일 암호화")


if __name__ == "__main__":
    migrate(sys.argv[1:])
