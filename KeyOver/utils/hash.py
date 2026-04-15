import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


if __name__ == "__main__":
    password = "Emilio123"
    password_hash = hash_password(password)
    print(password_hash)