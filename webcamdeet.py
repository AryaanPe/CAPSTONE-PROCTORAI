import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import threading
import cv2
import numpy as np
import pandas as pd
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load the machine learning models
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
model = load_model('bestnew.h5')

# Initialize constants
LAPLACIAN_THRESH = 20
GAUSSIAN_THRESH = 15
WEIGHTED_SCORE_COEFF_LAPLACIAN = 0.25
WEIGHTED_SCORE_COEFF_GAUSSIAN = 0.75
SCORE_THRESHOLD = 50

# Initialize global variables for webcam processing
cap = cv2.VideoCapture(0)
reference_image_path = None  # Reference image path input from user
reference_face_encoding = None
questions_df = pd.read_excel('excelTEST.xlsx')  # Load questions from the Excel file
current_question_index = 0
question_attempted = [False] * len(questions_df)
lock = threading.Lock()

# Function to encode image for displaying in the web app
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{frame_base64}"

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div(id='camera-feed', style={'height': '300px', 'border': '1px solid black'}),
            dcc.Upload(id='upload-image', children=html.Button('Upload Reference Image')),
            html.Div(id='reference-image-display')
        ], width=4),
        dbc.Col([
            html.Div([
                html.H4("Question"),
                html.Div(id='question-text', style={'margin-bottom': '10px'}),
                dcc.Checklist(id='mcq-options', options=[], value=[])
            ], style={'border': '1px solid black', 'padding': '10px', 'margin-bottom': '20px'}),
            html.Div(id='question-navigation', children=[
                dbc.Button('Next Question', id='next-question', color='primary', style={'margin': '5px'}),
                dbc.Button('Submit', id='submit', color='success', style={'margin': '5px'})
            ]),
            html.Div(id='all-questions', style={'margin-top': '20px'}),
        ], width=8)
    ])
])

# Function to load reference image
def load_reference_image(path):
    global reference_face_encoding
    image = face_recognition.load_image_file(path)
    reference_face_locations = face_recognition.face_locations(image)
    reference_face_encodings = face_recognition.face_encodings(image, reference_face_locations)

    if len(reference_face_encodings) > 0:
        reference_face_encoding = reference_face_encodings[0]
        return encode_image(image)
    else:
        return None

# Callback for camera feed display
@app.callback(Output('camera-feed', 'children'), Input('next-question', 'n_clicks'))
def update_camera_feed(n_clicks):
    ret, frame = cap.read()
    if ret:
        # Here, perform all the overlays (face detection, live/spoof detection, etc.) and show the processed frame
        # Add face matching, artifact detection logic and more here
        return html.Img(src=encode_image(frame), style={'height': '100%', 'width': '100%'})
    return None

# Callback for reference image upload
@app.callback(Output('reference-image-display', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def display_reference_image(contents, filename):
    if contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        path = f'/mnt/data/{filename}'
        with open(path, 'wb') as f:
            f.write(decoded)
        ref_image_display = load_reference_image(path)
        if ref_image_display:
            return html.Img(src=ref_image_display, style={'height': '200px', 'width': '200px'})
    return None

# Callback to update questions
@app.callback(
    [Output('question-text', 'children'), Output('mcq-options', 'options'), Output('all-questions', 'children')],
    [Input('next-question', 'n_clicks'), Input('submit', 'n_clicks')],
    State('mcq-options', 'value')
)
def update_question(next_clicks, submit_clicks, selected_options):
    global current_question_index, question_attempted

    # Check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", [], None
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Navigation logic
    if button_id == 'next-question':
        current_question_index += 1
    elif button_id == 'submit' and current_question_index < len(questions_df):
        question_attempted[current_question_index] = True

    # Ensure question index is within bounds
    current_question_index = max(0, min(current_question_index, len(questions_df) - 1))

    # Update question and options
    question = questions_df.iloc[current_question_index]['Question']
    options = [{'label': opt, 'value': opt} for opt in questions_df.iloc[current_question_index][1:].dropna().values]

    # Display all questions for navigation
    question_buttons = [
        dbc.Button(
            str(i + 1), 
            id=f'question-{i}', 
            color='success' if question_attempted[i] else 'danger', 
            style={'margin': '5px'}
        ) 
        for i in range(len(questions_df))
    ]

    return question, options, question_buttons

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
