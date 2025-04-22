from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from Levenshtein import distance as levenshtein_distance
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import random

# Inisialisasi Flask dan database
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/sistem_penilaian'
db = SQLAlchemy(app)

# Model database
class Soal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pertanyaan = db.Column(db.String(500))
    kunci_jawaban = db.Column(db.String(500))

class Jawaban(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama_siswa = db.Column(db.String(100))
    soal_id = db.Column(db.Integer)
    jawaban = db.Column(db.String(500))
    skor = db.Column(db.Float)

# Routing halaman siswa
@app.route("/")
def siswa():
    soal_semua = Soal.query.all()
    soal_acak = random.sample(soal_semua, 5)  # ambil acak 5 soal
    return render_template("siswa.html", soal=soal_acak)


# Load BERT
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Fungsi Penilaian Semantik
def hitung_semantik(jawaban_siswa, jawaban_benar):
    with torch.no_grad():
        inputs_siswa = tokenizer(jawaban_siswa, return_tensors="pt", padding=True, truncation=True)
        inputs_benar = tokenizer(jawaban_benar, return_tensors="pt", padding=True, truncation=True)
        emb_siswa = model(**inputs_siswa).last_hidden_state.mean(dim=1)
        emb_benar = model(**inputs_benar).last_hidden_state.mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(emb_siswa, emb_benar)
        return similarity.item()

# Fungsi Penilaian Sintaksis
def hitung_sintaksis(jawaban_siswa, jawaban_benar):
    jarak = levenshtein_distance(jawaban_siswa.lower(), jawaban_benar.lower())
    max_len = max(len(jawaban_siswa), len(jawaban_benar))
    if max_len == 0:
        return 1.0
    return 1.0 - (jarak / max_len)

# Routing halaman siswa
@app.route("/")
def siswa():
    soal = Soal.query.all()
    return render_template("siswa.html", soal=soal)

# Proses submit jawaban
@app.route("/submit", methods=["POST"])
def submit():
    nama_siswa = request.form.get("nama_siswa")
    hasil_nilai = 0

    for soal in Soal.query.all():
        jawaban_siswa = request.form.get(f"jawaban_{soal.id}")
        jawaban_benar = soal.kunci_jawaban

        skor_semantik = hitung_semantik(jawaban_siswa, jawaban_benar)
        skor_sintaksis = hitung_sintaksis(jawaban_siswa, jawaban_benar)
        total_skor = (0.7 * skor_semantik + 0.3 * skor_sintaksis) * 100

        hasil_nilai += total_skor

        simpan_jawaban = Jawaban(
            nama_siswa=nama_siswa,
            soal_id=soal.id,
            jawaban=jawaban_siswa,
            skor=total_skor
        )
        db.session.add(simpan_jawaban)

    db.session.commit()
    return f"Selamat {nama_siswa}, nilai kamu: {hasil_nilai / len(Soal.query.all()):.2f}"

if __name__ == "__main__":
    app.run(debug=True)
