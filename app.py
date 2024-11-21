from flask import Flask, render_template, request,session, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import joblib
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# Load the model and scaler
model = joblib.load("model.pkl")  # Replace with your model file
scaler = joblib.load("scaler.pkl")  # Replace with your scaler file

# Constants for the calculation
BASE_PRICE = 165300
BASE_INDEX_VALUE = 100

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')

data = pd.read_csv('merged_data.csv', parse_dates=['DATE'])

#visualizations

@app.route('/visualizations')
def visualizations():
    # Select columns to plot
    indicators = ['DATE', 'CPIAUCSL', 'CSUSHPISA', 'GDP']
    selected_data = data[indicators]
    
    # Create a line chart using Plotly
    fig = go.Figure()
    for col in indicators[1:]:  # Skip 'DATE' column for x-axis
        fig.add_trace(go.Scatter(x=selected_data['DATE'], y=selected_data[col], mode='lines', name=col))

    # Update layout for better presentation
    fig.update_layout(
        title='Time Series of Key Economic Indicators',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x'
    )

    # Render the Plotly figure as JSON for HTML
    graph_json = pio.to_json(fig)
    return render_template('visualizations.html', graph_json=graph_json)


@app.route('/correlation-heatmap')
def correlation_heatmap():
    # Select numerical columns only for correlation
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    if 'DATE' in numerical_data.columns:
        numerical_data = numerical_data.drop(columns=['DATE'])
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    # Create heatmap with Plotly Express
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',  # Display correlations with two decimal points
        aspect="auto",
        color_continuous_scale="RdBu",  # Color scheme with diverging colors for positive/negative correlations
        labels=dict(color="Correlation"),  # Label the color scale
    )
    
    # Update layout for better readability and style
    fig.update_layout(
        title='Correlation Heatmap of Economic Indicators',
        title_x=0.5,  # Center the title
        font=dict(size=14),  # Increase font size for readability
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis=dict(tickangle=45, side="bottom"),  # Rotate x-axis labels and position them at the bottom
        yaxis=dict(tickangle=0, side="left"),
        margin=dict(l=60, r=60, t=60, b=60),  # Add padding around the heatmap
        coloraxis_colorbar=dict(
            title="Correlation",
            thickness=15,
            len=0.7,
            tickvals=[-1, -0.5, 0, 0.5, 1],  # Display important ticks only
        ),
    )
    
    # Convert figure to JSON for rendering in HTML
    graph_json = pio.to_json(fig)
    
    return render_template('visualizations.html', graph_json=graph_json)

@app.route('/distribution')
def distribution():
    # Choose a single column for distribution (e.g., CSUSHPISA)
    fig = px.histogram(data, x="CSUSHPISA", nbins=30, color_discrete_sequence=["blue"])
    fig.update_layout(title='Distribution of House Price Index (CSUSHPISA)', xaxis_title='CSUSHPISA', yaxis_title='Frequency')

    graph_json = pio.to_json(fig)
    return render_template('visualizations.html', graph_json=graph_json)

@app.route('/scatter-matrix')
def scatter_matrix():
    selected_data = data[['CPIAUCSL', 'CSUSHPISA', 'GDP']]  # Select relevant indicators
    fig = px.scatter_matrix(selected_data, dimensions=['CPIAUCSL', 'CSUSHPISA', 'GDP'], color="CSUSHPISA")
    fig.update_layout(title='Scatter Matrix of Selected Indicators')

    graph_json = pio.to_json(fig)
    return render_template('visualizations.html', graph_json=graph_json)


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html',user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

@app.route("/")
def home():
    return render_template("index.html")

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html')
# Route for the Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Collect input values from the form
    inputs = {
        "DATE": request.form["date"],
        "CPIAUCSL": float(request.form["cpi"]),
        "DAXRNSA": float(request.form["dax"]),
        "GDP": float(request.form["gdp"]),
        "HOUST": float(request.form["hous"]),
        "LXXRNSA": float(request.form["lxx"]),
        "MIXRNSA": float(request.form["mix"]),
        "MORTGAGE30US": float(request.form["mortgage"]),
        "NYXRSA": float(request.form["nyx"]),
        "PERMIT": float(request.form["permit"]),
        "POPTHM": float(request.form["pop"]),
        "SFXRNSA": float(request.form["sfx"]),
        "UNRATE": float(request.form["unrate"]),
        "year": float(request.form["year"]),
        "month": float(request.form["month"]),
        "CSUSHPISA_LAG1": float(request.form["chushpisa_lag1"]),
    }

    # Convert inputs to a DataFrame
    input_df = pd.DataFrame([inputs])
    input_df["DATE"] = pd.to_datetime(input_df["DATE"]).astype("int64") // 10**9

    # Preprocess and predict
    input_scaled = scaler.transform(input_df)
    predicted_csushpisa = model.predict(input_scaled)[0]

    # Calculate estimated price
    estimated_price = BASE_PRICE * (predicted_csushpisa / BASE_INDEX_VALUE)
    
    # Format prediction and price
    prediction_text = f"Predicted CSUSHPISA: {predicted_csushpisa:.2f}"
    estimated_price_text = f"Estimated House Price: ${estimated_price:,.2f}"

    return render_template("index.html", prediction_text=prediction_text, estimated_price_text=estimated_price_text)

    
    
if __name__ == "__main__":
    app.run(debug=True)
