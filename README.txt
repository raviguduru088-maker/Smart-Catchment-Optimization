SMART CATCHMENT OPTIMIZATION - DEMO PROJECT

1. Open this folder in VS Code.
2. Activate your Python 3.11 virtual environment (or create one).
3. Install: pip install -r requirements.txt
4. From this folder run:
   python src/train_model.py
   python src/run_analysis.py
   streamlit run website.py

Important:
- Dataset is synthetic (60 rows).
- Coordinates are synthetic.
- ML labels are synthetic; test accuracy is not real-world validation.
- SCS-CN soil/HSG mapping and CN lookup are provisional demo assumptions.
- MCDA weights are illustrative, not stakeholder validated.
- No live rainfall API/sensor feed is implemented.
