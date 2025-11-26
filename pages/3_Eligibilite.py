# -*- coding: utf-8 -*-
"""
Page 3 : Éligibilité - Interface Professionnelle
Revue approfondie des articles SCREENED_IN
"""
import streamlit as st
from src.database import get_db, Article
from src.eligibility_manager import EligibilityManager
import json

st.set_page_config(page_title="Éligibilité", layout="wide")

# Initialisation
db = next(get_db())
manager = EligibilityManager(db)

# ==================== HEADER ====================
st.title("📋 Phase Éligibilité - Revue Texte Complet")
st.caption("Évaluation finale des articles retenus au screening")

# ==================== STATISTIQUES ====================
stats = manager.get_review_stats()

col_prog, col_inc, col_exc, col_rem = st.columns(4)
col_prog.metric("Progression", f"{stats['progress_pct']:.1f}%")
col_inc.metric("✅ Inclus", stats['included'])
col_exc.metric("❌ Exclus", stats['excluded'])
col_rem.metric("📝 À réviser", stats['remaining'])

# Barre de progression
if stats['total_to_review'] + stats['reviewed'] > 0:
    total = stats['total_to_review'] + stats['reviewed']
    progress = stats['reviewed'] / total
    st.progress(progress)

st.divider()

# ==================== BODY ====================

# Vérifier s'il y a des articles
articles_to_review = manager.get_articles_to_review(limit=1)

if not articles_to_review:
    st.success("🎉 **Tous les articles ont été révisés !**")
    st.info("Passez à l'étape suivante : Analyse & Rapport")
    
    # Afficher statistiques finales
    with st.expander("📊 Statistiques finales", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total révisés", stats['reviewed'])
            st.metric("Taux d'inclusion", f"{(stats['included'] / stats['reviewed'] * 100):.1f}%" if stats['reviewed'] > 0 else "N/A")
        
        with col2:
            # Raisons d'exclusion
            exclusion_stats = manager.get_exclusion_stats()
            if exclusion_stats:
                st.write("**Distribution des raisons d'exclusion :**")
                for reason, count in sorted(exclusion_stats.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- {reason}: {count}")
    
    db.close()
    st.stop()

# Récupérer l'article courant
article = articles_to_review[0]

# ==================== LAYOUT SPLIT VIEW ====================

# Sidebar : Critères & Actions
with st.sidebar:
    st.subheader("🎯 Critères d'Éligibilité")
    
    criteria = manager.get_active_criteria()
    
    # Grouper par type
    inclusion = [c for c in criteria if c['type'] == 'INCLUSION']
    exclusion = [c for c in criteria if c['type'] == 'EXCLUSION']
    
    if inclusion:
        st.write("**✅ Critères d'inclusion:**")
        for c in inclusion:
            with st.expander(c['label'], expanded=False):
                st.caption(c['description'])
    
    if exclusion:
        st.write("**❌ Critères d'exclusion:**")
        for c in exclusion:
            with st.expander(c['label'], expanded=False):
                st.caption(c['description'])
    
    st.divider()
    
    # Navigation
    st.write(f"**Article {stats['reviewed'] + 1} / {stats['total_to_review'] + stats['reviewed']}**")
    
    if st.button("⏭️ Passer (sans décision)", use_container_width=True):
        st.warning("Article passé (restera en SCREENED_IN)")
        st.rerun()

# Main Content : Article Viewer
st.subheader(f"📄 {article.title}")

# Métadonnées
col_meta1, col_meta2, col_meta3 = st.columns(3)
col_meta1.caption(f"**Auteurs:** {article.authors or 'N/A'}")
col_meta2.caption(f"**Année:** {article.year or 'N/A'}")
col_meta3.caption(f"**Source:** {article.source or 'N/A'}")

if article.doi:
    st.caption(f"**DOI:** {article.doi}")

if article.link:
    st.markdown(f"🔗 [**Voir l'article complet en ligne**]({article.link})")

st.divider()

# Onglets : Abstract | Full Text
tab_abstract, tab_fulltext = st.tabs(["📝 Abstract", "📄 Texte Complet"])

with tab_abstract:
    if article.abstract:
        st.markdown(article.abstract)
    else:
        st.warning("Abstract non disponible")

with tab_fulltext:
    if article.full_text:
        # Chunking pour affichage
        text_length = len(article.full_text)
        
        if text_length > 5000:
            st.info(f"Texte complet : {text_length:,} caractères - Affiché en chunks")
            
            # Diviser en chunks
            chunk_size = 3000
            chunks = [article.full_text[i:i+chunk_size] for i in range(0, len(article.full_text), chunk_size)]
            
            chunk_selector = st.selectbox(
                "Section",
                range(len(chunks)),
                format_func=lambda x: f"Partie {x+1}/{len(chunks)}"
            )
            
            st.text_area(
                "Contenu",
                chunks[chunk_selector],
                height=400,
                disabled=True
            )
        else:
            st.text_area(
                "Texte complet",
                article.full_text,
                height=400,
                disabled=True
            )
    else:
        st.warning("⚠️ Texte complet non disponible")
        st.caption("Vous pouvez exclure pour cette raison ou consulter le lien externe")

st.divider()

# ==================== DÉCISION PANEL ====================

st.subheader("✍️ Décision")

col_decision, col_reasons = st.columns([1, 2])

with col_decision:
    decision = st.radio(
        "Statut final",
        ["INCLUDED", "EXCLUDED_ELIGIBILITY"],
        format_func=lambda x: "✅ Inclure" if x == "INCLUDED" else "❌ Exclure",
        key="decision_radio"
    )

with col_reasons:
    if decision == "EXCLUDED_ELIGIBILITY":
        # Raisons d'exclusion (multiselect)
        exclusion_labels = [c['label'] for c in exclusion]
        selected_reasons = st.multiselect(
            "Raisons d'exclusion (sélectionner toutes applicables)",
            exclusion_labels,
            key="exclusion_reasons"
        )
        
        if not selected_reasons:
            st.warning("⚠️ Veuillez sélectionner au moins une raison")
    else:
        selected_reasons = []
        st.success("Article sera inclus dans l'analyse finale")

# Notes (optionnel)
notes = st.text_area(
    "Notes du reviewer (optionnel)",
    placeholder="Commentaires, réserves, points à vérifier...",
    height=100,
    key="reviewer_notes"
)

# Bouton Soumettre
col_submit1, col_submit2 = st.columns([3, 1])

with col_submit2:
    can_submit = True
    if decision == "EXCLUDED_ELIGIBILITY" and not selected_reasons:
        can_submit = False
    
    if st.button("💾 Enregistrer Décision", 
                 type="primary", 
                 use_container_width=True,
                 disabled=not can_submit):
        
        success = manager.save_decision(
            article_id=article.id,
            decision=decision,
            reasons=selected_reasons,
            notes=notes,
            reviewer="User"  # TODO: ajouter gestion multi-reviewers
        )
        
        if success:
            st.success("✅ Décision enregistrée !")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Erreur lors de l'enregistrement")

with col_submit1:
    if not can_submit:
        st.error("⚠️ Sélectionnez au moins une raison d'exclusion")

db.close()
