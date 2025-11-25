import streamlit as st
import graphviz
from src.database import get_db, Article
from sqlalchemy import func

st.set_page_config(page_title="Analyse & PRISMA", page_icon="📊")

st.title("📊 Dashboard PRISMA")
st.markdown("Visualisation du flux de sélection des articles.")

db = next(get_db())

# Statistics
total_identified = db.query(Article).count()
screened_in = db.query(Article).filter(Article.status == "SCREENED_IN").count()
excluded_screening = db.query(Article).filter(Article.status == "EXCLUDED_SCREENING").count()
eligible_count = db.query(Article).filter(Article.status.in_(["SCREENED_IN", "ELIGIBLE", "INCLUDED", "EXCLUDED_ELIGIBILITY"])).count()
excluded_eligibility = db.query(Article).filter(Article.status == "EXCLUDED_ELIGIBILITY").count()
included = db.query(Article).filter(Article.status == "INCLUDED").count()

# PRISMA Diagram
dot = graphviz.Digraph(comment='PRISMA Flow Diagram')
dot.attr(rankdir='TB', size='8,8')
dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

# Nodes
dot.node('ID', f'Identification\nRecords identified: {total_identified}')
dot.node('SCREEN', f'Screening\nRecords screened: {total_identified}')
dot.node('EX_SCREEN', f'Records excluded\n(Title/Abstract): {excluded_screening}', fillcolor='salmon')
dot.node('ELIG', f'Eligibility\nFull-text assessed: {eligible_count}')
dot.node('EX_ELIG', f'Records excluded\n(Full-text): {excluded_eligibility}', fillcolor='salmon')
dot.node('INCL', f'Included\nStudies included: {included}', fillcolor='lightgreen')

# Edges
dot.edge('ID', 'SCREEN')
dot.edge('SCREEN', 'EX_SCREEN')
dot.edge('SCREEN', 'ELIG')
dot.edge('ELIG', 'EX_ELIG')
dot.edge('ELIG', 'INCL')

st.graphviz_chart(dot)

# Detailed Stats
st.divider()
st.subheader("Articles Inclus")
included_articles = db.query(Article).filter(Article.status == "INCLUDED").all()

if included_articles:
    for art in included_articles:
        st.write(f"- **{art.title}** ({art.year if art.year else 'N/A'}) - {art.authors}")
else:
    st.info("Aucun article inclus pour le moment.")

db.close()
