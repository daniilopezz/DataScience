import hashlib

password = "Emilio123"
password_hash = hashlib.sha256(password.encode()).hexdigest()
print(password_hash)