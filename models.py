from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(10), default='user') # 'admin' or 'user'

class PlantDisease(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_name = db.Column(db.String(100), nullable=False)
    disease_key = db.Column(db.String(100), nullable=False)
    cause = db.Column(db.Text, nullable=False)
    treatment = db.Column(db.Text, nullable=False)