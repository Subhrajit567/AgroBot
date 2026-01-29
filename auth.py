from flask import Blueprint, render_template, request, redirect, session, url_for

auth_bp = Blueprint('auth', __name__)

# Hardcoded creds (you can later replace with DB)
USERS = {
    "user": {"password": "user123", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"}
}

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USERS and USERS[username]["password"] == password:
            session['user'] = username
            session['role'] = USERS[username]["role"]

            if session['role'] == 'admin':
                return redirect('/admin')
            return redirect('/')

        return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
