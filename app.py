from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
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
import re
import pandas as pd
import io
import os
from werkzeug.utils import secure_filename

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
        # Get all answers for this user with their questions
        jawaban_detail = Jawaban.query.filter_by(id_user=session['user_id']).all()
        
        # Calculate average score
        total_nilai = sum(j.skor_akhir for j in jawaban_detail)
        nilai_akhir = total_nilai / len(jawaban_detail) if jawaban_detail else 0
        status = "Lulus" if nilai_akhir >= 75 else "Tidak Lulus"
        
        return render_template("siswa.html", 
                             show_result=True, 
                             nilai_akhir=nilai_akhir, 
                             status=status,
                             jawaban_detail=jawaban_detail)
    
    soal_semua = Soal.query.all()
    if len(soal_semua) < 10:
        flash('Minimal harus ada 10 soal yang tersedia', 'error')
        return redirect(url_for('login'))
    
    soal_acak = random.sample(soal_semua, 10)  # ambil acak 10 soal
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

# Import/Export routes
@app.route("/admin/soal/export")
@login_required
@role_required(['admin'])
def export_soal():
    try:
        # Query all soal from database
        soal_all = Soal.query.all()
        
        # Create DataFrame
        data = {
            'ID': [soal.id for soal in soal_all],
            'Pertanyaan': [soal.pertanyaan for soal in soal_all],
            'Kunci Jawaban': [soal.kunci_jawaban for soal in soal_all]
        }
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Soal', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Soal']
            
            # Format headers (bold)
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            
            # Set column widths
            worksheet.column_dimensions['A'].width = 8  # ID column
            worksheet.column_dimensions['B'].width = 50  # Pertanyaan column
            worksheet.column_dimensions['C'].width = 50  # Kunci Jawaban column
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='soal_export.xlsx'
        )
        
    except Exception as e:
        flash(f'Terjadi kesalahan saat mengekspor soal: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route("/admin/soal/template")
@login_required
@role_required(['admin'])
def download_template():
    try:
        # Create sample DataFrame
        data = {
            'ID': [1, 2],
            'Pertanyaan': [
                'Contoh pertanyaan 1?',
                'Contoh pertanyaan 2?'
            ],
            'Kunci Jawaban': [
                'Contoh jawaban untuk pertanyaan 1.',
                'Contoh jawaban untuk pertanyaan 2.'
            ]
        }
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Template', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Template']
            
            # Format headers (bold)
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            
            # Set column widths
            worksheet.column_dimensions['A'].width = 8  # ID column
            worksheet.column_dimensions['B'].width = 50  # Pertanyaan column
            worksheet.column_dimensions['C'].width = 50  # Kunci Jawaban column
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='template_soal.xlsx'
        )
        
    except Exception as e:
        flash(f'Terjadi kesalahan saat mengunduh template: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route("/admin/soal/import", methods=['POST'])
@login_required
@role_required(['admin'])
def import_soal():
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah', 'error')
        return redirect(url_for('admin_dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if not file.filename.endswith('.xlsx'):
        flash('File harus berformat Excel (.xlsx)', 'error')
        return redirect(url_for('admin_dashboard'))
    
    try:
        # Read Excel file
        df = pd.read_excel(file)
        
        # Validate required columns
        required_columns = ['Pertanyaan', 'Kunci Jawaban']
        if not all(col in df.columns for col in required_columns):
            flash('Format file tidak sesuai. Gunakan template yang disediakan', 'error')
            return redirect(url_for('admin_dashboard'))
        
        # Import questions
        success_count = 0
        skip_count = 0
        
        for _, row in df.iterrows():
            try:
                # Check if question already exists
                existing_soal = Soal.query.filter_by(
                    pertanyaan=row['Pertanyaan'],
                    kunci_jawaban=row['Kunci Jawaban']
                ).first()
                
                if existing_soal:
                    skip_count += 1
                    continue
                
                soal = Soal(
                    pertanyaan=row['Pertanyaan'],
                    kunci_jawaban=row['Kunci Jawaban']
                )
                db.session.add(soal)
                success_count += 1
            except Exception as e:
                continue
        
        db.session.commit()
        
        if success_count > 0:
            if skip_count > 0:
                flash(f'Berhasil mengimpor {success_count} soal baru. {skip_count} soal dilewati karena sudah ada.', 'success')
            else:
                flash(f'Berhasil mengimpor {success_count} soal baru.', 'success')
        else:
            if skip_count > 0:
                flash(f'Tidak ada soal baru yang diimpor. {skip_count} soal dilewati karena sudah ada.', 'info')
            else:
                flash('Tidak ada soal yang berhasil diimpor', 'error')
        
    except Exception as e:
        flash(f'Terjadi kesalahan saat mengimpor soal: {str(e)}', 'error')
    
    return redirect(url_for('admin_dashboard'))

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

# Load BERT
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
model.eval()  # Set model to evaluation mode

# Cache for storing embeddings
embedding_cache = {}

# Fungsi Penilaian Semantik dengan caching
def hitung_semantik(jawaban_siswa, jawaban_benar):
    # Check cache first
    cache_key = f"{jawaban_siswa}_{jawaban_benar}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    with torch.no_grad():
        # Batch process both inputs together
        inputs = tokenizer([jawaban_siswa, jawaban_benar], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(outputs[0].unsqueeze(0), outputs[1].unsqueeze(0))
        score = similarity.item()
        
        # Cache the result
        embedding_cache[cache_key] = score
        return score

# Fungsi Penilaian Sintaksis yang dioptimasi
def hitung_sintaksis(jawaban_siswa, jawaban_benar):
    """
    Menghitung skor sintaksis dengan pendekatan yang lebih efisien.
    """
    # Preprocessing yang lebih ringan
    def preprocess(text):
        return ' '.join(text.lower().split())

    # Preprocess input
    jawaban_siswa = preprocess(jawaban_siswa)
    jawaban_benar = preprocess(jawaban_benar)
    
    # Split words once
    kata_siswa = set(jawaban_siswa.split())
    kata_benar = set(jawaban_benar.split())
    
    # Quick length check
    if not kata_benar or not kata_siswa:
        return 0.0
    
    # Calculate word match score (50%)
    kata_cocok = kata_siswa.intersection(kata_benar)
    word_match_score = len(kata_cocok) / len(kata_benar)
    
    # Calculate sequence score (30%)
    sequence_score = SequenceMatcher(None, jawaban_siswa, jawaban_benar).quick_ratio()
    
    # Calculate length score (20%)
    length_ratio = min(len(jawaban_siswa), len(jawaban_benar)) / max(len(jawaban_siswa), len(jawaban_benar))
    
    # Calculate final score
    skor = (word_match_score * 0.5) + (sequence_score * 0.3) + (length_ratio * 0.2)
    
    # Add bonus for very similar answers
    if word_match_score > 0.8 and sequence_score > 0.7:
        skor = min(1.0, skor + 0.1)
    
    return skor

# Proses submit jawaban
@app.route("/submit", methods=["POST"])
@login_required
@role_required(['siswa'])
def submit():
    submitted_question_ids = [int(key.split('_')[1]) for key in request.form.keys() if key.startswith('jawaban_')]
    
    if len(submitted_question_ids) != 10:
        flash('Error: Jumlah soal tidak sesuai', 'error')
        return redirect(url_for('siswa'))
    
    soal_ujian = Soal.query.filter(Soal.id.in_(submitted_question_ids)).all()
    total_nilai = 0
    
    # Delete existing answers
    Jawaban.query.filter_by(id_user=session['user_id']).delete()
    db.session.commit()
    
    # Process answers in batches
    jawaban_batch = []
    for soal in soal_ujian:
        jawaban_siswa = request.form.get(f"jawaban_{soal.id}")
        if jawaban_siswa:
            # Calculate scores
            skor_semantik = hitung_semantik(jawaban_siswa, soal.kunci_jawaban)
            skor_sintaksis = hitung_sintaksis(jawaban_siswa, soal.kunci_jawaban)
            
            # Calculate final score (80% semantik, 20% sintaksis)
            skor_akhir = (0.8 * skor_semantik + 0.2 * skor_sintaksis) * 100
            
            # Apply bonus
            if 60 <= skor_akhir < 75:
                if 70 <= skor_akhir < 75:
                    skor_akhir += 7
                elif 65 <= skor_akhir < 70:
                    skor_akhir += 5
                else:
                    skor_akhir += 3
            
            skor_akhir = min(100, skor_akhir)
            status = "Lulus" if skor_akhir >= 75 else "Tidak Lulus"
            
            jawaban_batch.append(Jawaban(
                id_user=session['user_id'],
                id_soal=soal.id,
                jawaban_siswa=jawaban_siswa,
                skor_semantik=skor_semantik * 100,
                skor_sintaksis=skor_sintaksis * 100,
                skor_akhir=skor_akhir,
                status_akhir=status
            ))
            total_nilai += skor_akhir
    
    # Batch insert answers
    db.session.bulk_save_objects(jawaban_batch)
    db.session.commit()
    
    nilai_akhir = total_nilai / len(soal_ujian)
    status = "Lulus" if nilai_akhir >= 75 else "Tidak Lulus"
    
    # Get answer details
    jawaban_detail = (Jawaban.query
        .filter_by(id_user=session['user_id'])
        .join(Soal)
        .add_columns(
            Soal.pertanyaan,
            Soal.kunci_jawaban,
            Jawaban.jawaban_siswa,
            Jawaban.skor_semantik,
            Jawaban.skor_sintaksis,
            Jawaban.skor_akhir,
            Jawaban.status_akhir
        )
        .all())
    
    return render_template(
        "siswa.html", 
        show_result=True, 
        nilai_akhir=nilai_akhir, 
        status=status,
        jawaban_detail=jawaban_detail
    )

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True, port=8080)
