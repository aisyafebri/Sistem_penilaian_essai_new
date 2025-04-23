from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from Levenshtein import distance as levenshtein_distance
from transformers import BertTokenizer, BertModel
from functools import wraps
import torch
import numpy as np
import random
from difflib import SequenceMatcher
from collections import Counter
import nltk

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Inisialisasi Flask dan database
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/sistem_penilaian'
db = SQLAlchemy(app)

# Model database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class Soal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pertanyaan = db.Column(db.String(500))
    kunci_jawaban = db.Column(db.String(500))

class Jawaban(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    id_user = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    id_soal = db.Column(db.Integer, db.ForeignKey('soal.id'), nullable=False)
    jawaban_siswa = db.Column(db.String(500))
    skor_semantik = db.Column(db.Float)
    skor_sintaksis = db.Column(db.Float)
    skor_akhir = db.Column(db.Float)
    status_akhir = db.Column(db.String(50))

    # Relationships
    user = db.relationship('User', backref='jawaban')
    soal = db.relationship('Soal', backref='jawaban')

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Role-based access decorator
def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session:
                flash('Silakan login terlebih dahulu')
                return redirect(url_for('login'))
            if session['user_role'] not in allowed_roles:
                flash('Anda tidak memiliki akses ke halaman ini')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Login route
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:  # In production, use proper password hashing
            session['user_id'] = user.id
            session['user_role'] = user.role
            session['username'] = user.username
            
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('siswa'))
        else:
            flash('Username atau password salah')
            return redirect(url_for('login'))
    
    return render_template("login.html")

# Logout route
@app.route("/logout")
def logout():
    session.clear()
    flash('Anda telah berhasil logout')
    return redirect(url_for('login'))

# Routing halaman siswa
@app.route("/siswa")
@login_required
@role_required(['siswa'])
def siswa():
    # Check if there are existing answers for this user
    jawaban_exists = Jawaban.query.filter_by(id_user=session['user_id']).first()
    if jawaban_exists:
        # Calculate average score and status
        jawaban_user = Jawaban.query.filter_by(id_user=session['user_id']).all()
        total_nilai = sum(j.skor_akhir for j in jawaban_user)
        nilai_akhir = total_nilai / len(jawaban_user) if jawaban_user else 0
        status = "Lulus" if nilai_akhir >= 75 else "Tidak Lulus"
        return render_template("siswa.html", show_result=True, nilai_akhir=nilai_akhir, status=status)
    
    soal_semua = Soal.query.all()
    soal_acak = random.sample(soal_semua, 5)  # ambil acak 5 soal
    return render_template("siswa.html", soal=soal_acak, show_result=False)

# Admin dashboard route
@app.route("/admin/dashboard")
@login_required
@role_required(['admin'])
def admin_dashboard():
    jawaban_semua = Jawaban.query.all()
    soal_semua = Soal.query.all()
    users = User.query.all()
    return render_template("admin_dashboard.html", 
                         jawaban=jawaban_semua, 
                         soal=soal_semua,
                         users=users)

# Kelola Soal routes
@app.route("/admin/soal")
@login_required
@role_required(['admin'])
def kelola_soal():
    soal_semua = Soal.query.all()
    return render_template("kelola_soal.html", soal=soal_semua)

@app.route("/admin/soal/tambah", methods=['POST'])
@login_required
@role_required(['admin'])
def tambah_soal():
    pertanyaan = request.form.get('pertanyaan')
    kunci_jawaban = request.form.get('kunci_jawaban')
    
    if pertanyaan and kunci_jawaban:
        soal_baru = Soal(
            pertanyaan=pertanyaan,
            kunci_jawaban=kunci_jawaban
        )
        db.session.add(soal_baru)
        db.session.commit()
        flash('Soal berhasil ditambahkan', 'success')
    else:
        flash('Pertanyaan dan kunci jawaban harus diisi', 'error')
    
    return redirect(url_for('kelola_soal'))

@app.route("/admin/soal/edit/<int:id>", methods=['POST'])
@login_required
@role_required(['admin'])
def edit_soal(id):
    soal = Soal.query.get_or_404(id)
    soal.pertanyaan = request.form.get('pertanyaan')
    soal.kunci_jawaban = request.form.get('kunci_jawaban')
    db.session.commit()
    flash('Soal berhasil diperbarui', 'success')
    return redirect(url_for('kelola_soal'))

@app.route("/admin/soal/hapus/<int:id>")
@login_required
@role_required(['admin'])
def hapus_soal(id):
    soal = Soal.query.get_or_404(id)
    db.session.delete(soal)
    db.session.commit()
    flash('Soal berhasil dihapus', 'success')
    return redirect(url_for('kelola_soal'))

# Kelola User routes
@app.route("/admin/users")
@login_required
@role_required(['admin'])
def kelola_user():
    users = User.query.all()
    return render_template("kelola_user.html", users=users)

@app.route("/admin/users/tambah", methods=['POST'])
@login_required
@role_required(['admin'])
def tambah_user():
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role')
    
    if username and password and role:
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan', 'error')
        else:
            user_baru = User(
                username=username,
                password=password,  # In production, use password hashing
                role=role
            )
            db.session.add(user_baru)
            db.session.commit()
            flash('User berhasil ditambahkan', 'success')
    else:
        flash('Semua field harus diisi', 'error')
    
    return redirect(url_for('kelola_user'))

@app.route("/admin/users/edit/<int:id>", methods=['POST'])
@login_required
@role_required(['admin'])
def edit_user(id):
    user = User.query.get_or_404(id)
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role')
    
    if username and role:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user and existing_user.id != id:
            flash('Username sudah digunakan', 'error')
        else:
            user.username = username
            if password:  # Only update password if provided
                user.password = password  # In production, use password hashing
            user.role = role
            db.session.commit()
            flash('User berhasil diperbarui', 'success')
    else:
        flash('Username dan role harus diisi', 'error')
    
    return redirect(url_for('kelola_user'))

@app.route("/admin/users/hapus/<int:id>")
@login_required
@role_required(['admin'])
def hapus_user(id):
    if id == session['user_id']:
        flash('Tidak dapat menghapus akun sendiri', 'error')
    else:
        user = User.query.get_or_404(id)
        db.session.delete(user)
        db.session.commit()
        flash('User berhasil dihapus', 'success')
    return redirect(url_for('kelola_user'))

# Proses submit jawaban
@app.route("/submit", methods=["POST"])
@login_required
@role_required(['siswa'])
def submit():
    total_nilai = 0
    jumlah_soal = 0

    for soal in Soal.query.all():
        jawaban_siswa = request.form.get(f"jawaban_{soal.id}")
        if jawaban_siswa:
            # Ubah bobot: 70% sintaksis (Levenshtein), 30% semantik
            skor_semantik = hitung_semantik(jawaban_siswa, soal.kunci_jawaban)
            skor_sintaksis = hitung_sintaksis(jawaban_siswa, soal.kunci_jawaban)
            skor_akhir = (0.8 * skor_sintaksis + 0.2 * skor_semantik) * 100

            status = "Lulus" if skor_akhir >= 75 else "Tidak Lulus"
            
            simpan_jawaban = Jawaban(
                id_user=session['user_id'],
                id_soal=soal.id,
                jawaban_siswa=jawaban_siswa,
                skor_semantik=skor_semantik * 100,
                skor_sintaksis=skor_sintaksis * 100,
                skor_akhir=skor_akhir,
                status_akhir=status
            )
            db.session.add(simpan_jawaban)
            total_nilai += skor_akhir
            jumlah_soal += 1

    db.session.commit()
    nilai_akhir = total_nilai / jumlah_soal if jumlah_soal > 0 else 0
    status = "Lulus" if nilai_akhir >= 75 else "Tidak Lulus"
    
    return render_template("siswa.html", show_result=True, nilai_akhir=nilai_akhir, status=status)

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
    """
    Menghitung skor sintaksis menggunakan Levenshtein Distance murni,
    setelah teks dinormalisasi menggunakan NLP preprocessing.
    """
    def preprocess_nlp(text):
        # 1. Case folding
        text = text.lower()
        
        try:
            # 2. Tokenisasi kata menggunakan NLTK
            tokens = word_tokenize(text)
            
            # 3. Hapus stopwords
            stop_words = set(stopwords.words('indonesian'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            # Fallback jika NLTK gagal: tokenisasi sederhana
            print("NLTK resources not found, using simple tokenization")
            # Hapus tanda baca
            text = re.sub(r'[^\w\s]', ' ', text)
            # Tokenisasi sederhana dengan split
            tokens = text.split()
            # Stopwords manual untuk bahasa Indonesia
            stop_words = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan', 'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'tentang', 'seperti', 'ketika', 'bagi', 'sampai', 'karena', 'jika', 'namun', 'sehingga', 'yaitu', 'yakni', 'daripada', 'adalah'}
            tokens = [word for word in tokens if word not in stop_words]
        
        # 4. Normalisasi kata
        tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens]
        
        # 5. Filter token kosong
        tokens = [word for word in tokens if word]
        
        # 6. Urutkan kata (untuk konsistensi)
        tokens.sort()
        
        # 7. Gabung kembali menjadi kalimat
        return ' '.join(tokens)

    # Preprocessing dengan NLP
    jawaban_siswa_clean = preprocess_nlp(jawaban_siswa)
    jawaban_benar_clean = preprocess_nlp(jawaban_benar)

    # Debug print
    print("\nDetail Penilaian Sintaksis:")
    print(f"Jawaban setelah preprocessing NLP:")
    print(f"Kunci   : {jawaban_benar_clean}")
    print(f"Jawaban : {jawaban_siswa_clean}")

    # Hitung Levenshtein distance
    distance = levenshtein_distance(jawaban_siswa_clean, jawaban_benar_clean)
    max_length = max(len(jawaban_benar_clean), len(jawaban_siswa_clean))
    
    # Hitung skor (1 - normalized_distance)
    if max_length == 0:
        return 0.0
        
    skor = 1 - (distance / max_length)
    print(f"\nLevenshtein Distance: {distance}")
    print(f"Max Length: {max_length}")
    print(f"Skor Akhir: {skor:.2f}")
    
    return max(0.0, min(1.0, skor))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True, port=8080)
