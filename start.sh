#!/bin/bash
pip install -r requirements.txt
python app.py
flask run --host=0.0.0.0 --port=3000
