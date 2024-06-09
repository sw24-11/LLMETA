from app import app
import os

UPLOAD_FOLDER = 'uploads'
#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run()

#terminal
#cd ai
#waitress-serve --listen=0.0.0.0:8000 wsgi:app