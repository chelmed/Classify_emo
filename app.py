from flask import Flask, render_template, request
import face_recognition
import albumentations as A
import torch
from PIL import Image
import numpy as np

model = torch.jit.load('artifact/model_emo.pt')
model.eval()

emotion_id = {0: 'angry',
 1: 'disgusted',
 2: 'fearful',
 3: 'happy',
 4: 'neutral',
 5: 'sad',
 6: 'surprised'}

normalization = A.Normalize(
              mean=[0.485],
              std=[0.229],
              p=1.0
          )

def get_predict(img) -> str:
    """input object image (face), output class emotion"""
    img = Image.fromarray(img).convert('L')
    img = np.array(img.resize((96, 96)))
    img = normalization(image=img)['image']
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    logits = torch.argmax(model(tensor.unsqueeze(0))['logits'], dim=1)
    return emotion_id[logits.item()]


def get_emotions(path_img: str) -> str:
    """input path image and output all emotions on image"""
    detect_emo = []
    image = face_recognition.load_image_file(f"{path_img}")
    face_locations = face_recognition.face_locations(image)
    if len(face_locations)==0:
        return "Не обнаружены люди на картинке"
    else:
        for face in face_locations:
            top, right, bottom, left = face
            detect_emo.append(get_predict(image[top:bottom, left:right]))
    return 'Обнаружены люди с эмоциями: '  + ', '.join(detect_emo)


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		emo_img = get_emotions(img_path)
	return render_template("index.html", prediction=emo_img, img_path=img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
