from flask import Blueprint, render_template, request, jsonify, session
from symptom_db import plant_disease_db

admin_bp = Blueprint('admin', __name__)

def admin_only():
    return session.get('role') == 'admin'


# ------------------ ADMIN UI ------------------
@admin_bp.route('/admin')
def admin_panel():
    if not admin_only():
        return "Unauthorized", 403
    return render_template('admin.html')


# ------------------ ADMIN DATA API ------------------
@admin_bp.route('/admin/data')
def admin_data():
    if not admin_only():
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(plant_disease_db)


# ------------------ ADD / UPDATE ------------------
@admin_bp.route('/admin/add', methods=['POST'])
def add_disease():
    if not admin_only():
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    plant = data['plant']
    disease = data['disease']

    plant_disease_db.setdefault(plant, {})[disease] = {
        "cause": data['cause'],
        "treatment": data['treatment']
    }

    return jsonify({"status": "success"})


# ------------------ DELETE ------------------
@admin_bp.route('/admin/delete', methods=['POST'])
def delete_disease():
    if not admin_only():
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    plant = data['plant']
    disease = data['disease']

    if plant in plant_disease_db:
        plant_disease_db[plant].pop(disease, None)

    return jsonify({"status": "deleted"})
