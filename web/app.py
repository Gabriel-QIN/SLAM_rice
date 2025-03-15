import os
import re
import pandas as pd
import uuid
import sys
from flask import Flask, Blueprint, render_template, request, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import BooleanField, FieldList, FormField, TextAreaField, FileField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf import CSRFProtect
from werkzeug.utils import secure_filename
from flask_cors import CORS
SLAM_root_dir = '/public/softwares/SLAM_rice'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f'{SLAM_root_dir}/script')))
from predict import *

# slam_rice = Blueprint('slam_rice', __name__, url_prefix='/SLAM_rice')

app = Flask(__name__)
CORS(app)
# app.register_blueprint(slam_rice)
# app.url_map.strict_slashes = False
# app.config['APPLICATION_ROOT'] = '/SLAM_rice'
app.secret_key = 'MY KEY'  # Change this to a random secret key
app.template_folder = 'templates'
csrf = CSRFProtect(app)  # Enable CSRF protection

# Helper function to check if the sequence contains valid amino acids
def is_valid_amino_acid_sequence(sequence):
    # Standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    return bool(re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', sequence))

# Helper function to check if the file is in FASTA format
def is_fasta_format(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            if not first_line.startswith(">"):
                return False
            for line in file:
                if line.startswith(">"):
                    return False  # Lines after the first one should not start with '>'
            return True
    except Exception as e:
        return False

class PTMChoiceForm(FlaskForm):
    acetylation = BooleanField('Acetylation', default=True)
    crotonylation = BooleanField('Crotonylation', default=True)
    hydroxyisobutyrylation = BooleanField('2-Hydroxyisobutyrylation', default=True)
    malonylation = BooleanField('Malonylation', default=True)
    succinylation = BooleanField('Succinylation', default=True)
    lactylation = BooleanField('Lactylation', default=True)

class PredictionForm(FlaskForm):
    all_ptm = BooleanField('All', default=True)
    ptm_choices = FieldList(FormField(PTMChoiceForm), min_entries=1)
    sequences = TextAreaField('Enter sequence(s) in FASTA format:', render_kw={"placeholder": "Enter sequences in FASTA format here..."})
    sequence_file = FileField('Or upload a file in FASTA format:')
    submit = SubmitField('Submit')

    # Add cutoff options
    all_cutoff = BooleanField('All', default=True)  # Select All Cutoff
    cutoff_high = BooleanField('High', default=True)
    cutoff_medium = BooleanField('Medium', default=True)
    cutoff_low = BooleanField('Low', default=True)

    def __init__(self, *args, **kwargs):
        super(PredictionForm, self).__init__(*args, **kwargs)
        # Initialize ptm_choices with one PTMChoiceForm instance
        if not self.ptm_choices.entries:
            self.ptm_choices.append_entry()

@app.route('/home')
def home():
    return render_template('home.html')  # Pass the form to the template

@app.route('/')
def direct():
    return render_template('home.html')  # Pass the form to the template

@app.route('/citation')
def citation():
    return render_template('citation.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form = PredictionForm()
        os.makedirs('static/results/raw', exist_ok=True)
        if form.validate_on_submit():
            task_id = str(uuid.uuid4())
            session['task_id'] = task_id
            # print(url_for('predict'), url_for('results', task_id=task_id))
            # Save uploaded file if exists
            sequence_file = form.sequence_file.data
            print(sequence_file)
            if sequence_file:
                if not (sequence_file.filename.lower().endswith(('.fa', '.fasta'))):
                    flash('Uploaded file is either not in valid FASTA format or contains invalid amino acid sequences.', 'error')
                    return redirect(url_for('predict'))
                file_path = os.path.join('uploads', secure_filename(sequence_file.filename))
                sequence_file.save(file_path)
                
                # Check if the uploaded file is a valid FASTA format
                if not is_fasta_format(file_path):
                    flash('Uploaded file is either not in valid FASTA format or contains invalid amino acid sequences.', 'error')
                    return redirect(url_for('predict'))

                seq_path = file_path
            else:
                # Save the sequences to a temporary file
                seq_path = f'static/results/raw/{task_id}.fasta'
                with open(seq_path, 'w') as f:
                    f.write(form.sequences.data)

                # Check if the entered sequences are valid
                sequences = [seq for seq in form.sequences.data.splitlines() if not seq.startswith('>')]
                for seq in sequences:
                    if not is_valid_amino_acid_sequence(seq.strip()):
                        flash('Input sequences are invalid. Only standard amino acids are allowed. Only FASTA format are allowed.', 'error')
                        return redirect(url_for('predict'))

            # Process selected PTMs
            selected_ptms = []
            if form.all_ptm.data:
                selected_ptms = ['Acetylation', 'Crotonylation', '2-Hydroxyisobutyrylation', 'Malonylation', 'Succinylation', 'Lactylation']
            else:
                if form.ptm_choices[0].acetylation.data:
                    selected_ptms.append('Acetylation')
                if form.ptm_choices[0].crotonylation.data:
                    selected_ptms.append('Crotonylation')
                if form.ptm_choices[0].hydroxyisobutyrylation.data:
                    selected_ptms.append('2-Hydroxyisobutyrylation')
                if form.ptm_choices[0].malonylation.data:
                    selected_ptms.append('Malonylation')
                if form.ptm_choices[0].succinylation.data:
                    selected_ptms.append('Succinylation')
                if form.ptm_choices[0].lactylation.data:
                    selected_ptms.append('Lactylation')

            # Handle cutoff options
            cutoff_options = []
            if form.all_cutoff.data:
                cutoff_options = ['High', 'Medium', 'Low']
            else:
                if form.cutoff_high.data:
                    cutoff_options.append('High')
                if form.cutoff_medium.data:
                    cutoff_options.append('Medium')
                if form.cutoff_low.data:
                    cutoff_options.append('Low')
            # print(cutoff_options)

            # Call your prediction logic here
            try:
                run_prediction_logic(seq_path, selected_ptms, cutoff_options, task_id)
                return redirect(url_for('results', task_id=task_id))  # Redirect to results page
            except Exception as e:
                flash(f'An error occurred during prediction: {str(e)}')
                return redirect(url_for('predict'))

        else:
            print("Form validation failed.")
            print(form.errors)  # Print form errors for debugging
        return render_template('loading.html', task_id=task_id)
    else:
        form = PredictionForm()
    return render_template('predict.html', form=form)

@app.route('/check_status/<task_id>')
def check_status(request, task_id):
    task_dir = os.path.join('static/results', task_id)
    result_file = os.path.join(task_dir, 'result.csv')
    if os.path.exists(result_file):
        return JsonResponse({'status': 'complete', 'task_id': task_id})
    else:
        return JsonResponse({'status': 'running'})

@app.route('/results/<task_id>')
def results(task_id):
    task_id = session.get('task_id')
    results_path = f'static/results/{task_id}/results.csv'
    all_results_path = f'static/results/{task_id}/all_results.csv'
    # pie_image_path = f'static/results/{task_id}/pie.png'
    # bar_image_path = f'static/results/{task_id}/bar.png'

    df = pd.read_csv(results_path, sep='\t')
    df_all = pd.read_csv(all_results_path, sep='\t')
    # styled_df = df.style.apply(highlight_row, axis=1)
    # styled_df.set_table_attributes('style="border-collapse: collapse; border: 1px solid black;"')
    # styled_df.set_table_styles({
    #     '': {
    #         'selector': 'th',
    #         'props': [('border', '1px solid black')]
    #     },
    #     '': {
    #         'selector': 'td',
    #         'props': [('border', '1px solid black')]
    #     }
    # })
    # prediction_data = styled_df.render()
    prediction_data = df.to_html(classes='table table-striped', index=False)  # 生成 HTML 表格
    # prediction_data_all = df_all.to_html(index=False, classes='dataframe')  # 生成 HTML 表格
    if os.path.exists(all_results_path):
        return render_template('results.html', task_id=task_id, prediction=prediction_data)
    else:
        return "Results not found", 404

@app.route('/retrieve')
def retrieve():
    return render_template('retrieve.html')

@app.route('/loading', methods=['GET', 'POST'])
def loading():
    task_id = session.get('task_id')
    return render_template('loading.html', task_id=task_id)

def run_prediction_logic(seq_path, selected_ptms, cutoff_options, task_id):
    pdict = {'kac':'Acetylation', 'kcr':'Crotonylation', 'kmal':'Malonylation', 'ksucc':'Succinylation', 'khib':'2-Hydroxyisobutyrylation', 'kla':'Lactylation'}
    abbr_dict = {v:k for k,v in pdict.items()}
    # Your prediction logic here
    chain = 'A'
    pdb_path = None
    mode = 'seq' if pdb_path is None else 'struct'
    
    # Prepare PTM types for prediction
    ptm_types = ','.join([abbr_dict[ptm].lower() for ptm in selected_ptms])  # Example: 'kac,kcr,khib,kmal,ksucc,kla'
    
    pred_df = []
    all_df = []
    threshold_list = []
    for ptm in ptm_types.split(','):
        threshold = threshold_data[mode][ptm]['Sp 80']  # Example threshold
        tholds = [threshold_data[mode][ptm]['Sp 95'], threshold_data[mode][ptm]['Sp 90'], threshold_data[mode][ptm]['Sp 85'], threshold]
        threshold_list.append(tholds)
        
        # Call your prediction engine (you need to implement this function)
        case, kbhb = predict_engine(seq_path, pdb_path, threshold=tholds, ptm=ptm, chain=chain, use_PLM=False)
        all_df.append(case)
        pred_df.append(kbhb)
    # Combine results into DataFrames
    all_df = pd.concat(all_df)
    # Generate plots (you need to implement these functions)
    os.makedirs(f'static/results/{task_id}', exist_ok=True)
    ptm_list = [p.replace('k', 'K') for p in ptm_types.split(',')]
    draw_pie(all_df, ptm_list, f'static/results/{task_id}/pie.png')
    draw_bar(all_df, ptm_list, f'static/results/{task_id}/bar.png')
    all_df.to_csv(f'static/results/{task_id}/all_results.csv', sep='\t', index=False)
    final_df = all_df[all_df['Confidence'].isin(cutoff_options)]
    # Save results to CSV
    final_df.to_csv(f'static/results/{task_id}/results.csv', sep='\t', index=False)
    



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=8001)
