# -*- coding: utf-8 -*-
"""
Page 4 : Analyse & Reporting - Interface Professionnelle
Dashboard PRISMA avec diagramme flow, statistiques et exports
"""
import streamlit as st
from src.database import get_db, Article
from src.analytics import (
    get_global_stats,
    get_exclusion_distribution,
    get_temporal_distribution,
    get_source_distribution,
    get_included_articles_summary
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.exporters import export_csv, export_json, export_bibtex, export_excel
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Analyse & Reporting", layout="wide")

# ==================== HEADER ====================
st.title("📊 Analyse & Reporting PRISMA")
st.caption("Synthèse complète de la revue systématique")

db = next(get_db())

# ==================== STATISTIQUES GLOBALES ====================
st.subheader("📈 Vue d'Ensemble")

stats = get_global_stats(db)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📥 Identifiés", stats['identified'])
col2.metric("❌ Exclus Screening", stats['screened_out'])
col3.metric("✅ Screenés IN", stats['screened_in'])
col4.metric("❌ Exclus Éligibilité", stats['excluded_eligibility'])
col5.metric("🎯 INCLUS", stats['included'], 
           delta=f"{stats['inclusion_rate']:.1f}%" if stats['identified'] > 0 else "N/A")

# Barre de progression globale
if stats['identified'] > 0:
    progress_pct = (stats['included'] + stats['screened_out'] + stats['excluded_eligibility']) / stats['identified']
    st.progress(progress_pct)
    st.caption(f"Progression : {progress_pct*100:.1f}% des articles traités")

st.divider()

# ==================== PRISMA FLOW DIAGRAM ====================
st.subheader("📊 PRISMA 2020 Flow Diagram")

col_diagram, col_info = st.columns([3, 1])

with col_diagram:
    # Générer diagramme PRISMA STANDARD
    with st.spinner("Génération du diagramme PRISMA..."):
        try:
            # Comptages
            total_identified = db.query(Article).count()
            
            if total_identified == 0:
                st.warning("⚠️ Aucun article en base de données. Lancez une recherche d'abord.")
            else:
                screened_completed = db.query(Article).filter(
                    Article.status.in_(['SCREENED_IN', 'SCREENED_OUT', 'INCLUDED', 'EXCLUDED_ELIGIBILITY'])
                ).count()
                screened_out = db.query(Article).filter(Article.status == 'SCREENED_OUT').count()
                screened_in = db.query(Article).filter(
                    Article.status.in_(['SCREENED_IN', 'INCLUDED', 'EXCLUDED_ELIGIBILITY'])
                ).count()
                eligibility_assessed = db.query(Article).filter(
                    Article.status.in_(['INCLUDED', 'EXCLUDED_ELIGIBILITY'])
                ).count()
                excluded_eligibility = db.query(Article).filter(
                    Article.status == 'EXCLUDED_ELIGIBILITY'
                ).count()
                included = db.query(Article).filter(Article.status == 'INCLUDED').count()
                
                # Figure
                fig, ax = plt.subplots(figsize=(11, 10))
                ax.set_xlim(0, 12)
                ax.set_ylim(0, 10)
                ax.axis('off')
                
                # Helper pour rectangles
                def add_rect(x, y, w, h, text, facecolor='#E8F4F8', edgecolor='#2C3E50'):
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=edgecolor, facecolor=facecolor)
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                           fontsize=10, weight='normal', wrap=True)
                
                # Helper flèches
                def add_arrow(x1, y1, x2, y2, style='->', color='black'):
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle=style, lw=2, color=color))
                
                # LABELS DES PHASES (gauche)
                phase_x = 0.3
                ax.text(phase_x, 8.7, 'Identification', fontsize=11, weight='bold', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5E8F7', edgecolor='black'))
                ax.text(phase_x, 4.7, 'Screening', fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5E8F7', edgecolor='black'))
                ax.text(phase_x, 2.7, 'Eligibility', fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5E8F7', edgecolor='black'))
                ax.text(phase_x, 0.7, 'Included', fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5E8F7', edgecolor='black'))
                
                # FLOW PRINCIPAL (centre) - Espacement uniforme de 2.0 unités
                y_start = 8.2
                spacing = 2.0  # Espacement uniforme
                
                # Identification
                add_rect(2, y_start, 4, 1, f'Records identified from\ndatabase searching\n(n={total_identified})')
                add_arrow(4, y_start, 4, y_start - 1.0)
                
                # Après déduplication
                y_dedup = y_start - spacing
                add_rect(2, y_dedup, 4, 1, f'Records after duplicates removed\n(n={total_identified})')
                add_arrow(4, y_dedup, 4, y_dedup - 1.0)
                
                # Screening
                y_screen = y_dedup - spacing
                add_rect(2, y_screen, 4, 1, f'Records screened\n(n={screened_completed})')
                add_arrow(4, y_screen, 4, y_screen - 1.0)
                
                # Exclusion Screening (droite)
                add_arrow(6, y_screen + 0.5, 7.5, y_screen + 0.5)
                add_rect(7.5, y_screen, 4, 1, f'Irrelevant records\nexcluded\n(n={screened_out})', 
                        facecolor='#FFE6E6')
                
                # Eligibility
                y_elig = y_screen - spacing
                add_rect(2, y_elig, 4, 1, f'Full-text articles assessed\nfor eligibility\n(n={eligibility_assessed})')
                add_arrow(4, y_elig, 4, y_elig - 1.0)
                
                # Exclusion Eligibility (droite)
                add_arrow(6, y_elig + 0.5, 7.5, y_elig + 0.5)
                add_rect(7.5, y_elig, 4, 1, f'Full-text articles\nexcluded\n(n={excluded_eligibility})',
                        facecolor='#FFE6E6')
                
                # Included
                y_inc = y_elig - spacing
                add_rect(2, y_inc, 4, 1, f'Studies included in\nanalysis\n(n={included})',
                        facecolor='#E6FFE6')
                
                # Titre
                ax.text(6, 9.5, 'PRISMA 2020 Flow Diagram', 
                       ha='center', fontsize=14, weight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            
        except Exception as e:
            st.error(f"❌ Erreur génération diagramme : {e}")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())

with col_info:
    st.write("**Taux de sélection :**")
    st.metric("Global", f"{stats['inclusion_rate']:.1f}%")
    
    if stats['identified'] > 0:
        screening_rate = (stats['screened_in'] / stats['identified'] * 100) if stats['identified'] > 0 else 0
        st.metric("Après Screening", f"{screening_rate:.1f}%")
    
    if stats['screened_in'] > 0:
        elig_rate = (stats['included'] / (stats['screened_in'] + stats['excluded_eligibility']) * 100)
        st.metric("Après Éligibilité", f"{elig_rate:.1f}%")

st.divider()

# ==================== VISUALISATIONS ====================
st.subheader("📉 Analyses Détaillées")

tab_exclusion, tab_temporal, tab_sources = st.tabs([
    "Raisons d'Exclusion",
    "Distribution Temporelle",
    "Sources"
])

with tab_exclusion:
    exclusion_dist = get_exclusion_distribution(db)
    
    col_screen, col_elig = st.columns(2)
    
    with col_screen:
        st.write("**Phase Screening**")
        if exclusion_dist['screening']:
            df_screen = pd.DataFrame(
                list(exclusion_dist['screening'].items()),
                columns=['Raison', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.bar_chart(df_screen.set_index('Raison'))
            st.dataframe(df_screen, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune exclusion screening")
    
    with col_elig:
        st.write("**Phase Éligibilité**")
        if exclusion_dist['eligibility']:
            df_elig = pd.DataFrame(
                list(exclusion_dist['eligibility'].items()),
                columns=['Raison', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.bar_chart(df_elig.set_index('Raison'))
            st.dataframe(df_elig, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune exclusion éligibilité")

with tab_temporal:
    temporal = get_temporal_distribution(db)
    
    if temporal['years']:
        df_temporal = pd.DataFrame({
            'Année': temporal['years'],
            'Nombre d\'articles': temporal['counts']
        })
        
        st.line_chart(df_temporal.set_index('Année'))
        st.dataframe(df_temporal, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de données temporelles")

with tab_sources:
    sources = get_source_distribution(db)
    
    if sources:
        df_sources = pd.DataFrame(
            list(sources.items()),
            columns=['Source', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.bar_chart(df_sources.set_index('Source'))
        st.dataframe(df_sources, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de données sources")

st.divider()

# ==================== ARTICLES INCLUS ====================
st.subheader(f"📋 Articles Inclus (n = {stats['included']})")

if stats['included'] > 0:
    included_summary = get_included_articles_summary(db)
    df_included = pd.DataFrame(included_summary)
    
    # Affichage tableau
    st.dataframe(
        df_included,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "ID",
            "title": st.column_config.TextColumn("Titre", width="large"),
            "authors": "Auteurs",
            "year": "Année",
            "source": "Source",
            "doi": "DOI",
            "link": st.column_config.LinkColumn("Lien"),
            "reviewer": "Reviewer",
            "reviewed_at": "Date Revue"
        }
    )
else:
    st.info("Aucun article inclus. Complétez la phase Éligibilité.")

st.divider()

# ==================== EXPORTS ====================
st.subheader("💾 Exports")

col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

with col_exp1:
    if st.button("📄 Export CSV", use_container_width=True):
        try:
            path = export_csv()
            if path:
                st.success(f"✅ Exporté : {Path(path).name}")
                with open(path, 'rb') as f:
                    st.download_button(
                        "⬇️ Télécharger CSV",
                        f,
                        file_name=Path(path).name,
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"Erreur export CSV : {e}")

with col_exp2:
    if st.button("📊 Export Excel", use_container_width=True):
        try:
            path = export_excel()
            if path:
                st.success(f"✅ Exporté : {Path(path).name}")
                with open(path, 'rb') as f:
                    st.download_button(
                        "⬇️ Télécharger Excel",
                        f,
                        file_name=Path(path).name,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
        except Exception as e:
            st.error(f"Erreur export Excel : {e}")

with col_exp3:
    if st.button("📝 Export JSON", use_container_width=True):
        try:
            path = export_json()
            if path:
                st.success(f"✅ Exporté : {Path(path).name}")
                with open(path, 'rb') as f:
                    st.download_button(
                        "⬇️ Télécharger JSON",
                        f,
                        file_name=Path(path).name,
                        mime='application/json'
                    )
        except Exception as e:
            st.error(f"Erreur export JSON : {e}")

with col_exp4:
    if st.button("📚 Export BibTeX", use_container_width=True):
        try:
            path = export_bibtex()
            if path:
                st.success(f"✅ Exporté : {Path(path).name}")
                with open(path, 'rb') as f:
                    st.download_button(
                        "⬇️ Télécharger BibTeX",
                        f,
                        file_name=Path(path).name,
                        mime='application/x-bibtex'
                    )
        except Exception as e:
            st.error(f"Erreur export BibTeX : {e}")

# Info exports
with st.expander("ℹ️ À propos des exports"):
    st.markdown("""
    **Formats disponibles :**
    - **CSV** : Compatible Excel, Google Sheets
    - **Excel** : Format .xlsx avec mise en forme
    - **JSON** : Données structurées pour analyse
    - **BibTeX** : Citations pour LaTeX, Mendeley, Zotero
    
    Les fichiers sont sauvegardés dans `data/` et téléchargeables directement.
    """)

db.close()
