# -*- coding: utf-8 -*-
"""
Script Rapide pour Générer les Visualisations
==============================================

Génère automatiquement tous les graphiques après l'exécution du pipeline.

Usage:
    python generate_visualizations.py
"""

import os
import sys

def main():
    print("\n" + "="*70)
    print("  GENERATION AUTOMATIQUE DES VISUALISATIONS")
    print("="*70 + "\n")
    
    # Vérifier les fichiers nécessaires
    report_file = "articles_report.json"
    csv_file = "articles_final.csv"
    
    if not os.path.exists(report_file):
        print(f"[ERREUR] Fichier rapport introuvable: {report_file}")
        print("\nExecutez d'abord le pipeline:")
        print("  python run_improved.py")
        print("\nOu si vous utilisez le pipeline original:")
        print("  python main.py")
        return 1
    
    print(f"[OK] Rapport trouve: {report_file}")
    
    if os.path.exists(csv_file):
        print(f"[OK] CSV trouve: {csv_file}")
    else:
        print(f"[WARN] CSV non trouve: {csv_file}")
        print("      Certaines visualisations seront limitees")
    
    print("\nChargement du module de visualisation...")
    
    try:
        from visualize import PipelineVisualizer
        
        output_dir = "visualizations"
        print(f"\nGeneration des graphiques dans: {output_dir}/\n")
        
        visualizer = PipelineVisualizer(
            report_path=report_file,
            csv_path=csv_file if os.path.exists(csv_file) else None
        )
        
        visualizer.generate_all(output_dir)
        
        print("\n[SUCCESS] Toutes les visualisations ont ete generees !")
        print(f"\nConsultez le dossier '{output_dir}/' pour voir les graphiques.")
        
        return 0
        
    except ImportError as e:
        print(f"\n[ERREUR] Dependances manquantes: {e}")
        print("\nInstallez les dependances requises:")
        print("  pip install matplotlib seaborn scikit-learn")
        print("\nOu installez tout:")
        print("  pip install -r requirements_improved.txt")
        return 1
        
    except Exception as e:
        print(f"\n[ERREUR] Une erreur est survenue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

