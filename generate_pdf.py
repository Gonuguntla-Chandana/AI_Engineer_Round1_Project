"""
generate_pdf.py
Builds AI_Engineer_Round1_Project.pdf documenting the email classification solution.
"""

import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------------------------------------------------------------------
# 1. Load classified results (must exist already)
# -------------------------------------------------------------------
df = pd.read_csv("classified_emails_output.csv")   # output created by classify_emails.py

# -------------------------------------------------------------------
# 2. Configure PDF
# -------------------------------------------------------------------
pdf_path = "AI_Engineer_Round1_Project.pdf"
doc = SimpleDocTemplate(pdf_path)
styles = getSampleStyleSheet()
content = []

# -------------------------------------------------------------------
# 3. Title
# -------------------------------------------------------------------
content.append(Paragraph("AI Engineer Round 1 - Coding Assessment", styles['Title']))
content.append(Spacer(1, 18))

# -------------------------------------------------------------------
# 4. Problem Statement
# -------------------------------------------------------------------
problem = """
<b>Problem Statement:</b><br/>
Given a dataset of customer support emails, build a solution to classify
messages into categories and detect high-priority requests to improve
response efficiency.
"""
content.append(Paragraph(problem, styles['Normal']))
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 5. Approach
# -------------------------------------------------------------------
approach = """
<b>Approach & Methodology:</b><br/>
1. Combined email subject and body into a single feature.<br/>
2. Flagged priority using keywords such as <i>urgent</i>, <i>immediate</i>, <i>error</i>.<br/>
3. Used TF-IDF vectorization and a Naive Bayes classifier for categories:
   Account Issue, Billing Issue, Subscription, Technical/Integration, General Query.<br/>
4. Exported predictions and key metrics for evaluation.
"""
content.append(Paragraph(approach, styles['Normal']))
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 6. Execution Steps
# -------------------------------------------------------------------
steps = """
<b>Execution Steps:</b><br/>
1. Install dependencies: <i>pip install pandas scikit-learn reportlab</i><br/>
2. Place <i>Sample_Support_Emails_Dataset.csv</i> and <i>classify_emails.py</i> in this folder.<br/>
3. Run: <i>python classify_emails.py</i> to produce <i>classified_emails_output.csv</i>.<br/>
4. Run this script: <i>python generate_pdf.py</i> to create the documentation PDF.<br/>
"""
content.append(Paragraph(steps, styles['Normal']))
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 7. Results Summary
# -------------------------------------------------------------------
summary = df['category'].value_counts().to_dict()
summary_text = "<b>Results Summary:</b><br/>" + "<br/>".join([f"{cat}: {cnt}" for cat, cnt in summary.items()])
content.append(Paragraph(summary_text, styles['Normal']))
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 8. Table Preview
# -------------------------------------------------------------------
preview = [df.columns.tolist()] + df.head(5).values.tolist()
table = Table(preview)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT')
]))
content.append(Paragraph("<b>Sample Classified Emails:</b>", styles['Normal']))
content.append(table)
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 9. Submission Notes
# -------------------------------------------------------------------
notes = """
<b>Submission Notes:</b><br/>
- Public GitHub repository with all code, dataset, README, and this PDF.<br/>
- Demo video showing execution and outputs.<br/>
- Documentation structured for quick evaluation.
"""
content.append(Paragraph(notes, styles['Normal']))
content.append(Spacer(1, 12))

# -------------------------------------------------------------------
# 10. Build PDF
# -------------------------------------------------------------------
doc.build(content)
print(f"✅ PDF generated at {pdf_path}")
