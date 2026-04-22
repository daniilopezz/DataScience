# Utilidad de hashing de contraseñas con SHA-256.
# Utilità di hashing delle password con SHA-256.

import hashlib


def hash_password(password: str) -> str:
    # Devuelve el hash SHA-256 de la contraseña en texto plano.
    # Restituisce l'hash SHA-256 della password in chiaro.
    return hashlib.sha256(password.encode()).hexdigest()


if __name__ == "__main__":
    for name, pwd in [("Matteo", "Matteo123"), ("Diego", "Diego123"), ("Emilio", "Emilio123")]:
        print(f"{name}: {hash_password(pwd)}")
